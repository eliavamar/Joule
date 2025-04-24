import { MessageParam } from "@anthropic-ai/sdk/resources/index.mjs"
import {
	AzureOpenAiChatClient,
	AzureOpenAiChatCompletionRequestMessage,
	AzureOpenAiChatCompletionRequestSystemMessage,
	AzureOpenAiChatCompletionRequestUserMessageContentPart,
	AzureOpenAiChatCompletionStreamChunkResponse,
	AzureOpenAiChatCompletionStreamResponse,
} from "@sap-ai-sdk/foundation-models"
import { ApiHandler } from ".."
import { ApiHandlerOptions, ModelInfo, sapAiCoreDefaultModelId, SapAiCoreModelId, sapAiCoreModels } from "../../shared/api"
import { ApiStream } from "../transform/stream"
import { withRetry } from "../retry"
import { convertToOpenAiMessages } from "../transform/openai-format"
import OpenAI from "openai"
import { devspace } from "@sap/bas-sdk"
import { BASLLMProxy } from "../../utils/ai-core/bas-llm-proxy"
import { Logger } from "../../services/logging/Logger"
import fs from "fs"
import path from "path"
import os from "os"

const AI_CORE_CREDS_FILENAME = "ai-core-creds.json"

type ModelType = { id: SapAiCoreModelId; info: ModelInfo }

export class SapAiCore implements ApiHandler {
	private options: ApiHandlerOptions
	private basLLMProxy: BASLLMProxy | undefined

	constructor(options: ApiHandlerOptions) {
		this.options = options
		this.setupAiCore()
	}

	/**
	 * Converts OpenAI message format to Azure OpenAI message format
	 * @param openAiMessages Messages in OpenAI format
	 * @returns Messages in Azure OpenAI format
	 */
	private convertToAICoreOpenAiMessages(
		openAiMessages: OpenAI.Chat.ChatCompletionMessageParam[],
	): AzureOpenAiChatCompletionRequestMessage[] {
		return openAiMessages.map((message) => {
			// Handle different role types
			switch (message.role) {
				case "system":
					return {
						role: "system",
						content: message.content,
					}
				case "user":
					// For user messages, handle content parts
					if (Array.isArray(message.content)) {
						// Convert content parts to Azure format
						const azureContent: AzureOpenAiChatCompletionRequestUserMessageContentPart[] = message.content
							.filter((part) => part.type === "text" || part.type === "image_url")
							.map((part) => {
								if (part.type === "text") {
									return {
										type: "text" as const,
										text: part.text,
									}
								} else if (part.type === "image_url") {
									return {
										type: "image_url" as const,
										image_url: {
											url: part.image_url.url,
											detail: part.image_url.detail || "auto",
										},
									}
								}
								// This should never happen due to the filter above
								throw new Error(`Unsupported content part type: ${(part as any).type}`)
							})

						return {
							role: "user",
							content: azureContent,
							name: message.name,
						}
					} else {
						// String content
						return {
							role: "user",
							content: message.content || "",
							name: message.name,
						}
					}
				case "assistant":
					return {
						role: "assistant",
						content: message.content,
						name: message.name,
						tool_calls: message.tool_calls,
					}
				case "tool":
					return {
						role: "tool",
						content: message.content,
						tool_call_id: message.tool_call_id,
					}
				case "function":
					return {
						role: "function",
						content: message.content,
						name: message.name || "function",
					}
				default:
					// For any other roles, convert to system message as fallback
					return {
						role: "system",
						content: message.content,
					}
			}
		})
	}

	@withRetry()
	async *createMessage(systemPrompt: string, messages: MessageParam[]): ApiStream {
		const model = this.getModel()
		const modelInfo = model.info

		// Convert to OpenAI format first
		const openAImessages = [...convertToOpenAiMessages(messages)]
		// Then convert to Azure OpenAI format
		const azureSystemMessages: AzureOpenAiChatCompletionRequestSystemMessage = {
			role: "system",
			content: systemPrompt,
		}
		const azureMessages = [azureSystemMessages, ...this.convertToAICoreOpenAiMessages(openAImessages)]

		if (process.env["AICORE_SERVICE_KEY"]) {
			yield* this.llmRequestAICore(model, modelInfo, azureMessages)
		} else {
			yield* this.llmRequestBASProxy(model, azureMessages)
		}
	}

	private async *llmRequestAICore(model: ModelType, modelInfo: ModelInfo, azureMessages: any[]): ApiStream {
		const chatClient = new AzureOpenAiChatClient(model.id.trim())
		const streamOptions =
			model.id === "o3-mini"
				? { max_completion_tokens: modelInfo.maxTokens, messages: azureMessages }
				: { max_tokens: modelInfo.maxTokens, messages: azureMessages }

		const response = await chatClient.stream(streamOptions)

		for await (const chunk of response.stream) {
			const delta = chunk.getDeltaContent()
			if (delta) {
				yield {
					type: "text",
					text: delta,
				}
			}
		}
	}

	private async *llmRequestBASProxy(model: ModelType, azureMessages: any[]): ApiStream {
		const payload = {
			messages: azureMessages,
			stream: true,
		}

		const response = await this.basLLMProxy?.requestCompletion(model.id, payload)

		if (!response) {
			return
		}

		const rl = require("readline").createInterface({
			input: response,
			crlfDelay: Infinity,
		})

		for await (const line of rl) {
			if (line.startsWith("data: ")) {
				const data = line.slice(6).trim()
				if (data === "[DONE]") {
					break
				}

				try {
					const parsed = JSON.parse(data)
					const delta = parsed?.choices?.[0]?.delta?.content
					if (delta) {
						yield {
							type: "text",
							text: delta,
						}
					}
				} catch (err) {
					console.error("Error parsing streamed chunk:", err)
				}
			}
		}
	}

	getModel(): { id: SapAiCoreModelId; info: ModelInfo } {
		const modeId = (this.options.sapAiCoreModelId as SapAiCoreModelId) || sapAiCoreDefaultModelId
		return {
			id: modeId,
			info: sapAiCoreModels[modeId],
		}
	}

	loadAiCoreCredentials(): string | undefined {
		const credsFilePath = path.join(os.homedir(), AI_CORE_CREDS_FILENAME)

		if (!fs.existsSync(credsFilePath)) {
			return undefined
		}

		const fileContents = fs.readFileSync(credsFilePath, "utf-8")
		const parsed = JSON.parse(fileContents)
		return JSON.stringify(parsed)
	}

	setupAiCore(): void {
		let creds: string | undefined

		try {
			creds = this.loadAiCoreCredentials()
		} catch (err) {
			Logger.log(`Failed to load AI Core credentials: ${err}`)
		}

		if (creds) {
			process.env["AICORE_SERVICE_KEY"] = creds
			Logger.log(`AI Core service credentials loaded successfully from file ~/${AI_CORE_CREDS_FILENAME}.`)
		} else if (devspace.isBuildCode()) {
			Logger.log("AI Core credentials missing. Falling back to BAS Proxy LLM AI Core setup.")
			this.basLLMProxy = new BASLLMProxy()
			this.basLLMProxy.getDeployments()
		} else {
			Logger.log(
				`AI Core setup failed. Please check the credentials file ~/${AI_CORE_CREDS_FILENAME} or ensure working in BAS BuildCode.`,
			)
		}
	}
}
