import { MessageParam } from "@anthropic-ai/sdk/resources/index.mjs"

import { OrchestrationClient, LlmModuleConfig, TemplatingModuleConfig, ChatMessages } from "@sap-ai-sdk/orchestration"

import { ApiHandler } from ".."
import { ApiHandlerOptions, ModelInfo, sapAiCoreDefaultModelId, SapAiCoreModelId, sapAiCoreModels } from "../../shared/api"
import { ApiStream } from "../transform/stream"
import { withRetry } from "../retry"

export class SapAiCore implements ApiHandler {
	private options: ApiHandlerOptions

	constructor(options: ApiHandlerOptions) {
		this.options = options
		this.setupAiCoreEnvVariable()
	}

	/**
	 * Main entry point called by the router. It forwards the conversation to the
	 * orchestration runtime and yields streamed chunks back to the caller.
	 */
	@withRetry()
	async *createMessage(systemPrompt: string, messages: MessageParam[]): ApiStream {
		const model = this.getModel()

		// Define the LLM to be used by the Orchestration pipeline
		const llm: LlmModuleConfig = {
			model_name: model.id,
			model_params: {
				stream_options: {
					include_usage: true,
				},
			},
		}

		// Simple template: just pass the system prompt.  If you need RAG you can
		// replace this with the grounding template variant.
		const templating: TemplatingModuleConfig = {
			template: [
				{
					role: "system",
					content: systemPrompt,
				},
			],
		}

		const orchestrationClient = new OrchestrationClient({ llm, templating })

		const response = await orchestrationClient.stream({
			messagesHistory: convertCoreMessageToSAPMessages(messages),
		})

		for await (const chunk of response.stream) {
			const delta = chunk.getDeltaContent()
			if (delta) {
				yield { type: "text", text: delta }
			}
		}
	}

	// ---------------------------------------------------------------------------
	// Helpers
	// ---------------------------------------------------------------------------

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

/**
 * Converts generic MessageParam[] coming from the front‑end into the structure
 * SAP AI Orchestration expects (ChatMessages).
 */
const convertCoreMessageToSAPMessages = (messages: MessageParam[]): ChatMessages => {
	return messages
		.map((message) => {
			// keep every role the SDK supports
			if (
				message.role === "user" ||
				message.role === "assistant" ||
				message.role === "system" ||
				message.role === "tool" ||
				message.role === "function"
			) {
				return {
					role: message.role,
					content:
						typeof message.content === "string"
							? message.content
							: message.content.map((m) => {
									if (m.type === "text") {
										return { ...m }
									} else if (m.type === "image") {
										return {
											type: "image_url",
											image_url: {
												url: (m as any).image_url?.url ?? (m as any).image ?? "",
												detail: (m as any).image_url?.detail ?? "auto",
											},
										}
									}
									throw new Error(`Unsupported message type: ${(m as any).type}`)
								}),
				}
			}
			return null
		})
		.filter((message) => message !== null) as ChatMessages
}
