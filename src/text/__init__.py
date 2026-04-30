from .model import ModelRegistry, LLaMAModel, OpenAIModel, AzureOpenAIModel, GigaChatModel

ModelRegistry.register("llama", LLaMAModel)
ModelRegistry.register("openai", OpenAIModel)
ModelRegistry.register("azure_openai", AzureOpenAIModel)
ModelRegistry.register("gigachat", GigaChatModel)
ModelRegistry.register("gigachat_max", GigaChatModel)