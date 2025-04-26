import { MessageParam } from "@anthropic-ai/sdk/resources/index.mjs"
import { CoreMessage } from "ai";

console.log("SAP AI Core: module loaded");
import {
	AzureOpenAiChatClient,
	AzureOpenAiChatCompletionRequestMessage,
	AzureOpenAiChatCompletionRequestSystemMessage,
	AzureOpenAiChatCompletionRequestUserMessageContentPart,
	AzureOpenAiChatCompletionStreamChunkResponse,
	AzureOpenAiChatCompletionStreamResponse,
} from "@sap-ai-sdk/foundation-models"

import { OrchestrationClient, LlmModuleConfig, TemplatingModuleConfig, ChatMessages } from '@sap-ai-sdk/orchestration';


import { ApiHandler } from ".."
import { ApiHandlerOptions, ModelInfo, sapAiCoreDefaultModelId, SapAiCoreModelId, sapAiCoreModels } from "../../shared/api"
import { ApiStream } from "../transform/stream"
import { withRetry } from "../retry"
import { convertToOpenAiMessages } from "../transform/openai-format"
import OpenAI from "openai"

export class SapAiCore implements ApiHandler {
	private options: ApiHandlerOptions

	constructor(options: ApiHandlerOptions) {
		this.options = options
		this.setupAiCoreEnvVariable()
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
		const chatClient = new AzureOpenAiChatClient(model.id.trim())

		const llm: LlmModuleConfig = {
			model_name: 'anthropic--claude-3.7-sonnet',
			model_params: { max_tokens: 10000, temperature: 0.1 }
		};

		const groundingTemplate = [
			{
				role: "system",
				content: systemPrompt,
			}
		];
		const templating: TemplatingModuleConfig = {
			template: groundingTemplate
		};


		try {
			const orchestrationClient = new OrchestrationClient({
				llm,
				templating
			});
			try {
				const response = await orchestrationClient.stream({
					messagesHistory: convertCoreMessageToSAPMessages(messages),
				});
				for await (const chunk of response.stream) {
					const delta = chunk.getDeltaContent()
					if (delta === null || delta === undefined) {
						continue
					}
					yield {
						type: "text",
						text: delta,
					}
				}
				// return response.stream.toContentStream();
			} catch (error: any) {
				throw error;
			}
		} catch (error: any) {
			throw error;
		}
		// //'o3-mini'
		// const openAImessages = [...convertToOpenAiMessages(messages)]
		//
		// // Then convert to Azure OpenAI format
		// const azureSystemMessages: AzureOpenAiChatCompletionRequestSystemMessage = {
		// 	role: "system",
		// 	content: systemPrompt,
		// }
		// const azureMessages = [azureSystemMessages, ...this.convertToAICoreOpenAiMessages(openAImessages)]
		//
		// let response
		// if (model.id === "o3-mini") {
		// 	response = await chatClient.stream(
		// 		{
		// 			max_completion_tokens: modelInfo.maxTokens,
		// 			messages: azureMessages,
		// 		},
		// 		undefined,
		// 		{
		// 			params: {
		// 				"api-version": "2024-12-01-preview",
		// 			},
		// 			maxContentLength: modelInfo.contextWindow,
		// 		},
		// 	)
		// } else {
		// 	response = await chatClient.stream(
		// 		{
		// 			max_tokens: modelInfo.maxTokens,
		// 			messages: azureMessages,
		// 		},
		// 		undefined,
		// 		{
		// 			maxContentLength: modelInfo.contextWindow,
		// 		},
		// 	)
		// }
		// // Use the Azure-compatible messages
		//
		// for await (const chunk of response.stream) {
		// 	const delta = chunk.getDeltaContent()
		// 	if (delta === null || delta === undefined) {
		// 		continue
		// 	}
		// 	yield {
		// 		type: "text",
		// 		text: delta,
		// 	}
		// }
	}

	getModel(): { id: SapAiCoreModelId; info: ModelInfo } {
		const modeId = (this.options.sapAiCoreModelId as SapAiCoreModelId) || sapAiCoreDefaultModelId
		return {
			id: modeId,
			info: sapAiCoreModels[modeId],
		}
	}

	setupAiCoreEnvVariable(): void {
		const aiCoreServiceCredentials = {
			clientid: this.options.sapClientid,
			clientsecret: this.options.sapClientsecret,
			url: this.options.sapAuthUrl,
			serviceurls: {
				AI_API_URL: this.options.sapApiUrl,
			},
		}
		process.env["AICORE_SERVICE_KEY"] = JSON.stringify(aiCoreServiceCredentials)
	}
}

const convertCoreMessageToSAPMessages = (messages: MessageParam[]): ChatMessages => {
	return messages
		.map((message) => {
			if (message.role === "user" || message.role === "assistant") {
				return {
					role: message.role,
					content:
						typeof message.content === "string"
							? message.content
							: message.content.map((m) => {
								if (m.type === "text") {
									return {
										...m,
									};
								} else if (m.type === "image") {
									return {
										type: "image_url",
										image_url: {
											url: m.type as string,
										},
									};
								} else {
									throw new Error(`Unsupported message type: ${m.type}`);
								}
							}),
				};
			} else {return null;}
		})
		.filter((message) => message !== null) as ChatMessages;
};
