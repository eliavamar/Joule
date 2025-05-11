import { MessageParam } from "@anthropic-ai/sdk/resources/index.mjs"

import { ScenarioApi } from "@sap-ai-sdk/ai-api"

import { OrchestrationClient, LlmModuleConfig, TemplatingModuleConfig, ChatMessages } from "@sap-ai-sdk/orchestration"

import { ApiHandler } from ".."
import { ApiHandlerOptions, ModelInfo, sapAiCoreDefaultModelId, SapAiCoreModelId, sapAiCoreModels } from "../../shared/api"
import { ApiStream } from "../transform/stream"
import { withRetry } from "../retry"

export class SapAiCore implements ApiHandler {
	private options: ApiHandlerOptions
	private modelsReady: Promise<void> // ← 1️⃣  keep the promise
	private formattedModels: Model[] = [] // ← 2️⃣  keep the data in-memory

	constructor(options: ApiHandlerOptions) {
		this.options = options
		this.setupAiCoreEnvVariable()
		this.modelsReady = this.getModelsFromSDK()
			.then((models) => {
				// save the array in memory
				this.formattedModels = models
			})
			.catch((err) => {
				console.error("model fetch failed:", err)
				this.formattedModels = [] // fail‐safe
			})
	}

	/**
	 * Main entry point called by the router. It forwards the conversation to the
	 * orchestration runtime and yields streamed chunks back to the caller.
	 */
	@withRetry()
	async *createMessage(systemPrompt: string, messages: MessageParam[]): ApiStream {
		await this.modelsReady

		const model = this.getModel()

		// Define the LLM to be used by the Orchestration pipeline
		const llm: LlmModuleConfig = {
			model_name: model.id,
			// add model_params only for Anthropic models
			...(model.id.includes("anthropic") && {
				model_params: { max_tokens: 4096 },
			}),
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
		let response
		const modelData = this.formattedModels.find((m) => m.id === model.id)

		let streaming = modelData?.streamingSupported ?? false

		if (streaming) {
			response = await orchestrationClient.stream({
				messagesHistory: convertMessageParamToSAPMessages(messages),
			})
			for await (const chunk of response.stream) {
				const delta = chunk.getDeltaContent()
				if (delta) {
					yield { type: "text", text: delta }
				}
			}
		} else {
			response = await orchestrationClient.chatCompletion({
				messagesHistory: convertMessageParamToSAPMessages(messages),
			})
			const delta = response.getContent()
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

	async getModelsFromSDK(): Promise<Model[]> {
		try {
			const models = await ScenarioApi.scenarioQueryModels("foundation-models", {
				"AI-Resource-Group": "default",
			}).execute()

			// Filter for models that have "orchestration" in their allowedScenarios
			const orchestrationModels = models.resources.filter(
				(model) =>
					model.allowedScenarios.some((scenario: any) => scenario.scenarioId === "orchestration") &&
					model.versions.some((version) => version.isLatest && version.capabilities?.includes("text-generation")),
			)

			// Map the filtered models to the required format
			const formattedModels = orchestrationModels
				.map((model) => {
					// Get the latest version (assuming isLatest: true indicates the latest version)
					const latestVersion = model.versions.find((v) => v.isLatest === true) || model.versions[0]
					return {
						id: model.model,
						provider: model.provider || "SAP",
						name: model.displayName || model.name,
						capabilities: latestVersion.capabilities || [],
						inputTypes: latestVersion.inputTypes || [],
						streamingSupported: model.model.includes("meta") ? false : latestVersion.streamingSupported,
						historySupported: !model.model.includes("abap"),
						imageRecognitionSupported: latestVersion.capabilities?.includes("image-recognition") || false,
					}
				})
				.sort((a, b) => a.provider.localeCompare(b.provider))

			return formattedModels
		} catch (error) {
			console.log("error: ", error)
			return [] // Return an empty array in case of an error
		}
	}
}
export interface Model {
	id: string
	name: string
	provider: string
	capabilities: string[]
	inputTypes: string[]
	streamingSupported: boolean
	historySupported: boolean
	imageRecognitionSupported: boolean
}

/**
 * Converts generic MessageParam[] coming from the front‑end into the structure
 * SAP AI Orchestration expects (ChatMessages).
 */
const convertMessageParamToSAPMessages = (messages: MessageParam[]): ChatMessages => {
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
