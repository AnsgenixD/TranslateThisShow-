# Broadcast STT & Translation Benchmark Report

Evaluation of the Apple Silicon MLX STT & LLM Translation pipeline across test scenarios against on-screen broadcast ground truth.

## Summary Metrics

| Test Scenario | Audio Duration | On-Screen Recall | Matched / Valid Cues | Median Start Delta | RTF (Speed) | Generated Cues |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **01_variety_interview** | 50.0s | **88.9%** | 24/27 | ±0.21s (46.2% <200ms) | **0.529x** | 26 lines |
| **02_variety_banter** | 50.0s | **78.8%** | 26/33 | ±0.185s (57.1% <200ms) | **0.411x** | 14 lines |

**Overall Mean On-Screen Recall**: `83.8%` | **Mean Real-Time Factor**: `0.470x realtime`

## Test Set Overview
1. **`01_variety_interview`** (50s): 《綜藝大熱門》 rapid Taiwanese Mandarin with reduplicated slang (`干干燥燥 粉粉嫩嫩`) and live interpreting anecdotes.
2. **`02_variety_banter`** (50s): 《綜藝大熱門》 high-energy multi-speaker comedic crosstalk with loud background laughter and music.

## Key Findings
- **High Verbatim Audio Coverage**: The pipeline successfully transcribes full conversational sentences that go beyond the on-screen headline graphics, achieving high on-screen cue recall.
- **Autonomous Glossary & Acronym Recovery**: DeepSeek closed-loop auditing correctly recovers English K-pop acronyms (`Super Junior`, `NCT`).
- **Sub-Realtime Throughput**: Runs at ~0.45x realtime entirely on Apple Silicon unified memory (MLX GPU), enabling rapid batch translation.