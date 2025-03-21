# Why this fork?

This is an effort at reworking the Memoripy project and make it easier to install and extend it.

I advocate for using local LLMs (that don't come from big corporations adamant on snooping on, and reusing your data to their benefits) and the context size there can easily become a problem, especially on lower-end machines. And that's before talking about the fatal flaw of local LLMs not remembering their past interactions from one session to the next.

Memoripy looked like a promising solution, however a few parts of the project opposed resistance.

I own a powerful smartphone, a Redmagic 9 Pro with 16GB of LPDDR4 RAM. It's still a relatively "lower-end" machine, all things considered, allowing a 4k context window and using an 8B Q4 model. It gets around 5tps for prompt processing, and 6tps for inference using koboldcpp, which I find respectable. It follows I tried using this project on it, but installing on Termux proved to be fraught with roadblocks, from the hard dependency versions leading to not compiling some of the packages properly. I have remedied those, and the project now works. However, I wasn't satisfied with it in its current state.

# Improvements

Here's an unordered list of improvements I have brought:

- **Logic for forgetfulness** leading to less memory consumption from heavily decayed memories lying around both in the memory store and in persistent storage eventually, reducing the size of the memory graph and semantic clusters in the process. If interactions aren't accessed and not persisted in long-term memory, they can be let go.

- **The faiss-cpu dependency is now optional** making installation easier. Set the USE_FAISS environment variable to use it. Do note it is not yet completely implemented, and it will raise exceptions.

- **The models and storage options now have their own folder** and the models have been separated in their own files.

- **The storage options now include DBMSs with SQLStorage** and the accompanying ORM allows for different "Owners" to have their own memories.

- **The "interaction" has now its own object class: InteractionData.** This simplifies a lot of code.

- **LLM responses can now be streamed.**

- **The OllamaChatModel class in particular** has been improved to include the Ollama model's SYSTEM prompt.

- **Prompt-building customization** for guiding how the LLM "remembers" the relevant and recent interactions.

- **Timestamps are now included with each interaction** to give the LLM a sense of time.

- **Custom prompts** that are chained before invoking the LLM.

# Future developments

- **Properly implement FAISS in.** This may take a while. The current model for implementation is to monkey-patch faiss-cpu in the MemoryStore class when new methods are implemented, instead of using "if" statements to check for USE_FAISS. Preferably, only the relevant parts of the non-FAISS code should be deported in new methods to be monkey-patched out with FAISS-enabled code.

- **Caching recurring results.** In the OllamaChatModel, the SYSTEM prompt should be cached somehow, and this would accelerate the processing significantly instead of rehashing the same SYSTEM prompt for every interaction.

- **Tool-calling in OllamaChatModel.** Perhaps other interfaces will also allow this.

- **The koboldai/koboldcpp interface.** This model interface is supposedly faster and has a lot more features than Ollama.

- **Allow MemoryStore to hold the memories of more than one "Owner"** and choose which one to "talk through" for each interaction.

- **Improve the LLM's sense of time.** The current timestamp implementation is rather hit-and-miss.
