This style guide is based on the provided data synthesis script and focuses on the patterns established for interacting with modern LLM SDKs and handling data generation pipelines.

## 1. Google GenAI SDK Usage
The codebase utilizes the newer `google-genai` library. 
*   **Client Initialization:** Instantiate the client via `genai.Client(api_key=...)`.
*   **Model Access:** Use the `client.models.generate_content` method rather than the legacy `GenerativeModel` objects.
*   **Structured Configuration:** Use `types.GenerateContentConfig` and `types.SafetySetting` for specifying parameters.
*   **Safety Overrides:** When synthesizing "negative" or "rejected" data for DPO, explicitly set safety thresholds to `BLOCK_NONE` to prevent API-side filtering of simulated errors.

## 2. LLM Orchestration & Robustness
*   **System Instructions:** Separate persona and constraints from the user prompt by using the `system_instruction` parameter in the config.
*   **Recursive Retries:** Implement a retry wrapper (`generate_with_retry`) that handles specific failure modes:
    *   **Rate Limits (429):** Sleep for 60 seconds.
    *   **Timeouts:** Sleep for 10 seconds.
    *   **Quality Checks:** Check `response.text` for existence and minimum length requirements.
*   **Dynamic Prompt Augmentation:** If a response fails quality checks (e.g., too short), append corrective instructions (e.g., `"(IMPORTANT: PLEASE WRITE MORE...)"`) to the prompt for the next retry attempt.

## 3. Script Structure & Configuration
*   **Constant Block:** Define configuration (API keys, model names, file paths) and large multi-line prompts at the top of the file in uppercase constants.
*   **Main Guard:** Encapsulate logic in a `main()` function and use the `if __name__ == "__main__":` entry point.
*   **Stateful Feedback:** Use clear terminal logging (emojis like 🚀, 📂, ⚠️, ✅) to indicate script progress, filtering status, and retry attempts.

## 4. Data Processing Patterns
*   **Dataset Filtering:** Prioritize data quality before generation by checking for minimum character counts (`len(text) > N`) and valid types.
*   **DPO JSONL Format:** Save preference data in a flat `.jsonl` format containing `prompt`, `chosen`, and `rejected` keys.
*   **Metadata Inclusion:** Include the reasoning or instruction used to generate the rejected sample (e.g., `error_type_simulated`) within the output record for better auditability.

## 5. Resource Management
*   **Safety Sleep:** Even when using `tqdm`, include a hard `time.sleep(1)` (or more for free tiers) within the loop to proactively manage rate limits.
*   **Graceful Failures:** Use try-except blocks around dataset loading and client setup to provide descriptive error messages rather than raw stack traces.