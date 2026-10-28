# Lines worth a human listen

The checks are measurements (Whisper transcripts, loudness, clipping); nobody has *heard* these
films. Every line below passed the checks, but an agent or the coordinator flagged it as worth an
ear: an unusual word, a name, or a tone the transcript hinted at. Times are approximate (see each
episode's .srt).

| Ep | Where | What to listen for |
|---|---|---|
| all | product names | "vLLM" (read "V L L M"), "SGLang" ("S G Lang"), "llama.cpp" ("llama C P P") come from the lexicon — do they sound natural? |
| all | "Owl" | the course's chatbot; a listener sometimes writes "OWL" |
| 5 | growth and measurement sections | "kibibytes" and "gibibytes" (Whisper wrote "kibbytes", "jibabyte"); "…takes sixteen" runs together |
| 2 | measured section and outro | "We ran Owl's model…" (a listener heard "Al's"); "The road is the memory bandwidth" ("Rode"); "Prefill versus Decode" in the outro |
| 1 | two clocks section | "The first clock is time to first token… plus any queue" (Whisper heard "timed to" and "cue"); "the hundred and twenty-eight thousand tokens" |
| 3 | prefill section and intro | "two thousand and forty-eight tokens" (92%: the checker read Whisper's "2048" as a year — fixed in the checker afterwards); "Prefill versus Decode" |
| 4 | cost section | dollar amounts ("three dollars ninety-nine", "seven dollars twenty", "Batch thirty-two costs thirty-four cents") — do they sound like prices?; "this Mac's ridge" (heard "max ridge") |
| 6 | hook and FP8 sections | "kibibytes" (Whisper writes "kibabytes"); "PagedAttention" in the outro (heard "Page Detention") |
| 7 | blocks section | "PagedAttention" said three times (one hook line was reworded after both listeners heard "page detention"); "vLLM" |
| 8 | pricing section | "So providers price cached input low: Anthropic bills cache reads at a tenth…" (Whisper writes "cashed"/"cash") |
| 9 | sinks section | "sink" / "sinks" (Whisper writes "sync" / "syncs"); "the bytes" ("bites") |
| 10 | engines line | "every major serving engine schedules this way" (reworded: "vLLM" was heard "VLOM" inside a list of names) |
| 11 | percentile lines | "ninety-ninth percentile"; "Sarathi Serve" (said as two words) |
| 13 | tokens-per-pass and measured sections | "Owl-mini drafts for Owl-8B" (Whisper heard "Al Minni"); "alpha point eight", "alpha point nine" — clear as Greek letters?; "grey" for rejected guesses |
| 12 | split and price sections | "DistServe" (Whisper writes "disk serve"/"distserv"); "Kimi" ("Kimmy"); "a hundred and twenty-eight kibibytes for every prompt token" (reordered after the voice skipped the number); "goodput" (heard "good put") |
| 15 | two memories and bytes sections | "twenty-six hundred times smaller" (Whisper writes "2,600"); "scratchpad"; "SRAM" said as "S RAM"; "FlashAttention-2 and -3" in the versions section |
| 14 | EAGLE and multi-token sections | "EAGLE-3" (Whisper writes "Eagle III"; the line stayed at 92%); "Its next version, EAGLE-2"; "DeepSeek's version three"; "Medusa" |
| 16 | fusion and graphs section | "The next fix, CUDA graphs, takes the CPU out of it" — the only check.sh FAIL left in the course so far: whisper-small hears "Kudographs" (twice, two takes), whisper-large "CUDAGraphs"; does "CUDA" sound clear before "graphs"?; "a hundred and thirty-two streaming multiprocessors" |
| 18 | FP8 and measured sections | "the five-and-two flavour" (E5M2; Whisper writes "5-in-2"); "Llama three point one"; "buy" vs "by"; "INT8" and "FP8" said many times |
| 17 | grammar and determinism sections | "XGrammar" (Whisper writes "Xgrammer"); "Feynman" ("fineman", kept at 98%); "logits"; at 4:30 it is the longest episode so far |
| 19 | ledger, floats and outro | "four and a quarter bits" (Whisper writes "4.25"); "MXFP4" / "NVFP4"; "two-to-four" pruning; kit outro "Mixture of Experts at Serve Time" (heard "Surf Time" on every take — likely Ep20's intro too) |
| 20 | intro and batch sections | kit intro "Mixture of Experts at Serve Time" (passed here at 100%, but Ep19's outro heard "Surf"); "The Mixtral model" ("mixed-roll", kept at 96%); "DeepSeek"; "seven point nine of the eight are lit" (reworded after the numbers ran together) |
| 21 | all-reduce and wires sections | "the step is called all-reduce" (reworded from "an all-reduce"); "InfiniBand"; "NVLink" and "PCIe" side by side; "KV cache" in the 8-bit line (reworded after "KB cache") |
| 24 | money lines throughout | spoken prices ("three dollars ninety-nine", "a dollar six per million output tokens" — kept at 92%, Whisper writes "$1.06"; "two thousand eight hundred seventy-two dollars eighty"); "providers bill those tokens at a tenth…" (reworded after "Anthropic" was heard "andropic") |
| 23 | ladder, fits and engines sections | "M5 Max" (whisper-small heard "N5" then "M5 NOx"); "this Mac's ceiling" (heard "max ceiling", kept at 94%); "forty gibibytes of cache" (kept at 97%); "llama.cpp" and "Ollama"; 4:32, the longest episode |
| 22 | router and multi-LoRA sections | "least loaded picks the shortest queue" (Whisper writes "cue", kept at 95%); "Each LoRA adapter" (reworded after "Allura"); "S-LoRA"; "vLLM, SGLang, TensorRT-LLM, llama.cpp and MLX" in one breath |
| 25 | what-to-do and rehearse sections | "Keep the queue short, and even the slowest student gets an answer" (Whisper writes "cue", kept at 94%); "By default, vLLM recomputes it"; "preempted" |
| 26 | quiet hours and the end card | "But she has a whole H100 to herself: a dollar fifteen…" (kept at 91%, Whisper writes "$1.15"); "forty times less than alone" ("a loan"); "eight K reservations held only a quarter as many" (re-voiced by the coordinator so it no longer contradicts Episode 7's fifty-two); "Thanks for watching!" |
