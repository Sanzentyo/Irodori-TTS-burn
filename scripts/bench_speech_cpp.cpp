// Standalone comparison harness for speech.cpp d23c49f, linked to its unmodified
// irodoritts library. See the benchmark report for the build command and pins.
// Usage: bench_speech_cpp MODEL.gguf INPUT_DIR OUTPUT_DIR REQUESTS WARMUPS DEVICE [encoded|tokens|oracle]
// INPUT_DIR is made by prepare_speech_cpp_comparison.py or, for oracle mode,
// prepare_synthesis_comparison.py (native RF/MeanFlow schedule). Timing excludes model
// loading and diagnostics; tokens mode includes text/caption encoding. It also
// includes all sampler transfers and final audio readback in both modes.

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.h"
#include "irodori-tts/codec.h"
#include "irodori-tts/layout.h"
#include "irodori-tts/sampler.h"
#include "irodori-tts/text-encoder.h"
#include "irodori-tts/speaker-encoder.h"
#include "npy.h"

using namespace irodori;
using Clock = std::chrono::steady_clock;

static void save(const std::filesystem::path & path, const std::vector<float> & values) {
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--list-devices") {
            for (auto * device : devices()) {
                std::cout << ggml_backend_dev_name(device) << ": " << ggml_backend_dev_description(device) << '\n';
            }
            return 0;
        }
        if (argc != 7 && argc != 8) throw std::runtime_error("expected MODEL INPUT_DIR OUTPUT_DIR REQUESTS WARMUPS DEVICE [encoded|tokens|oracle]");
        const std::string condition_mode = argc == 8 ? argv[7] : "encoded";
        const bool oracle_fixture = condition_mode == "oracle";
        if (condition_mode != "encoded" && condition_mode != "tokens" && !oracle_fixture) throw std::runtime_error("invalid condition mode");
        const bool encode_conditions = condition_mode != "encoded";
        const std::filesystem::path input(argv[2]), output(argv[3]);
        const int requests = std::stoi(argv[4]), warmups = std::stoi(argv[5]);
        if (warmups < 1 || requests <= warmups) throw std::runtime_error("need warmup and measured requests");
        if (!std::filesystem::create_directory(output)) throw std::runtime_error("output directory already exists");
        auto read = [&](const std::string & name) { return read_npy((input / (name + ".npy")).string()); };
        const Npy text = read("text"), caption = read("caption"), noise = read("noise"), times = read("times");
        const bool has_reference = std::filesystem::exists(input / "ref_latent.npy");
        if (has_reference && !oracle_fixture) throw std::runtime_error("reference latents require oracle mode");
        const Npy reference = has_reference ? read("ref_latent") : Npy{};
        const int frames = static_cast<int>(noise.shape.at(0));
        const int steps = static_cast<int>(times.f32.size()) - 1;
        const Conditions conditions{text.f32, static_cast<int>(text.shape.at(0)), {}, 0, {},
                                    caption.f32, static_cast<int>(caption.shape.at(0))};
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(init_backend(argv[6]), ggml_backend_free);
        std::cerr << "backend: " << ggml_backend_name(backend.get()) << '\n';
        std::cerr << "device: " << ggml_backend_dev_description(ggml_backend_get_device(backend.get())) << '\n';
        const ModelFile model(argv[1], backend.get(), model_layout, [encode_conditions, has_reference](const std::string & name) {
            return name.rfind("dit.", 0) == 0 || name.rfind("codec.dec.", 0) == 0 ||
                (has_reference && name.rfind("speaker.", 0) == 0) ||
                (encode_conditions && (name.rfind("text.", 0) == 0 || name.rfind("caption.", 0) == 0));
        });
        const Dit dit(model);
        if ((!oracle_fixture && dit.meanflow()) || noise.shape.at(1) != dit.latent_dim()) throw std::runtime_error("incompatible fixture/model geometry");
        Sampler sampler(dit, model, backend.get());
        Codec codec(model, backend.get());
        const TextEncoder encoder(model);
        const auto speaker_encoder = has_reference ? std::make_unique<SpeakerEncoder>(model) : nullptr;
        const Npy text_ids = encode_conditions ? read("text_ids") : Npy{};
        const Npy caption_ids = encode_conditions ? read("caption_ids") : Npy{};
        if (encode_conditions && (text_ids.i32.size() != static_cast<size_t>(conditions.text_tokens) ||
                                  caption_ids.i32.size() != static_cast<size_t>(conditions.caption_tokens))) {
            throw std::runtime_error("token and condition counts differ");
        }
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> condition_allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())), ggml_gallocr_free);
        auto condition_request = [&]() {
            if (!encode_conditions) return conditions;
            Graph graph;
            auto * text_state = encoder.build(graph, text_ids.i32, Condition::Text);
            auto * caption_state = caption_ids.i32.empty() ? nullptr : encoder.build(graph, caption_ids.i32, Condition::Caption);
            auto * speaker_state = speaker_encoder ? speaker_encoder->build(graph, reference.f32) : nullptr;
            graph.output(text_state);
            if (speaker_state) graph.output(speaker_state);
            if (caption_state) graph.output(caption_state);
            graph.compute(backend.get(), condition_allocator.get());
            return Conditions{Graph::read(text_state), conditions.text_tokens, speaker_state ? Graph::read(speaker_state) : std::vector<float>{}, speaker_state ? static_cast<int>(speaker_state->ne[1]) : 0, {},
                              caption_state ? Graph::read(caption_state) : std::vector<float>{}, conditions.caption_tokens};
        };
        Guidance settings = sampler.guidance();
        if (!dit.meanflow()) {
            settings.text = 3;
            settings.caption = conditions.caption_tokens ? (oracle_fixture ? 3 : 4) : 0;
            settings.speaker = has_reference ? 5 : 0;
            settings.min_t = 0.5;
            settings.max_t = 1;
        }
        const Guidance guidance = sampler.check(settings, steps, has_reference, conditions.caption_tokens > 0);
        save(output / "native_times.f32le", sampler.schedule(steps));
        auto decode = [&](const std::vector<float> & latent) {
            std::vector<float> audio;
            audio.reserve(static_cast<size_t>(frames) * codec.hop());
            codec.decode(latent, static_cast<int64_t>(frames) * codec.hop(),
                         [frames](const DecodeProgress &) { return frames; },
                         [&](const float * samples, size_t n) {
                             audio.insert(audio.end(), samples, samples + n);
                             return true;
                         });
            return audio;
        };
        std::ofstream report(output / "timings.json");
        report.exceptions(std::ios::badbit | std::ios::failbit);
        report << std::setprecision(12) << "{\"backend\":" << std::quoted(ggml_backend_name(backend.get()))
               << ",\"frames\":" << frames << ",\"steps\":" << steps << ",\"sample_rate\":" << codec.sample_rate()
               << ",\"condition_mode\":" << std::quoted(condition_mode)
               << ",\"timing_boundary\":\"resident request through host audio; whole codec decode\",\"samples\":[\n";
        for (int i = 0; i < requests; ++i) {
            ggml_backend_synchronize(backend.get());
            const auto start = Clock::now();
            const auto current_conditions = condition_request();
            const auto condition_end = Clock::now();
            const auto latent = sampler.sample(current_conditions, noise.f32, frames, steps, guidance);
            ggml_backend_synchronize(backend.get());
            const auto rf_end = Clock::now();
            const auto audio = decode(latent);
            ggml_backend_synchronize(backend.get());
            const auto end = Clock::now();
            const double rf = std::chrono::duration<double>(rf_end - start).count();
            const double total = std::chrono::duration<double>(end - start).count();
            const double condition_seconds = std::chrono::duration<double>(condition_end - start).count();
            report << (i ? ",\n" : "") << "{\"request\":" << i + 1
                   << ",\"warmup\":" << (i < warmups ? "true" : "false")
                   << ",\"condition_seconds\":" << condition_seconds
                   << ",\"rf_seconds\":" << rf << ",\"consumer_seconds\":" << total << '}';
            report.flush();
            std::cerr << "request " << i + 1 << ": RF " << rf << " s, audio " << total << " s\n";
            if (i == warmups) {
                save(output / "latent.f32le", latent);
                save(output / "audio.f32le", audio);
                save(output / "text_state.f32le", current_conditions.text);
                save(output / "caption_state.f32le", current_conditions.caption);
            }
        }
        report << "\n],\"stream_samples\":[\n";
        for (int i = 0; i < requests; ++i) {
            ggml_backend_synchronize(backend.get());
            const auto start = Clock::now();
            const auto current_conditions = condition_request();
            const auto latent = sampler.sample(current_conditions, noise.f32, frames, steps, guidance);
            double first_audio = -1;
            std::vector<float> streamed;
            codec.decode(latent, static_cast<int64_t>(frames) * codec.hop(),
                [](const DecodeProgress & p) { return static_cast<int>(std::min<int64_t>(p.frames_left, p.frames_done == 0 ? 12 : 17)); },
                [&](const float * samples, size_t n) {
                    if (first_audio < 0) first_audio = std::chrono::duration<double>(Clock::now() - start).count();
                    streamed.insert(streamed.end(), samples, samples + n);
                    return true;
                });
            const double total = std::chrono::duration<double>(Clock::now() - start).count();
            report << (i ? ",\n" : "") << "{\"request\":" << i + 1 << ",\"warmup\":" << (i < warmups ? "true" : "false")
                   << ",\"first_audio_seconds\":" << first_audio << ",\"consumer_seconds\":" << total << '}';
            report.flush();
            if (i == warmups) save(output / "streamed_audio.f32le", streamed);
        }
        report << "\n]}\n";
        report.close();
        // Isolate numerical error of each forward from accumulated sampler
        // error: supply the Rust run's input and exact timestep at every step.
        for (int i = 0; !oracle_fixture && i < steps; ++i) {
            const std::string index = (i < 10 ? "0" : "") + std::to_string(i);
            const Npy x = read("input_" + index);
            std::vector<float> branches;
            sampler.velocity(conditions, x.f32, frames, times.f32, i, guidance, &branches);
            save(output / ("forward_" + index + ".f32le"), branches);
        }
        save(output / "decode_reference.f32le", decode(read("reference_latent").f32));
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
