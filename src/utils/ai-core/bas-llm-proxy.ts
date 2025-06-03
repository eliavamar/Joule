import axios from "axios"
import { Logger } from "../../services/logging/Logger"
import { Readable } from "stream"

export const BAS_URL_ENV_NAME = "H2O_URL"
export const BAS_LLM_SERVICE_NAME = "llm"

function getLLMServiceUrl(): string {
	return `${process.env[BAS_URL_ENV_NAME]}/${BAS_LLM_SERVICE_NAME}/v2`
}

export class BASLLMProxy {
	private static deployments: any[] = []

	public async getDeployments(resourceGroup?: string): Promise<any[]> {
		if (!BASLLMProxy.deployments || BASLLMProxy.deployments.length === 0) {
			const apiUrl = getLLMServiceUrl()
			const accessUrl = `${apiUrl}/lm/deployments`
			const ai_resource_group = resourceGroup || "default"

			const response = await axios.get(accessUrl, {
				headers: {
					"Content-Type": "application/json",
					"AI-Resource-Group": ai_resource_group,
				},
			})
			BASLLMProxy.deployments = response.data && response.data.resources
		}
		return BASLLMProxy.deployments
	}

	public async getDeploymentUrl(modelName: string, modelVersion?: string, resourceGroup?: string): Promise<string> {
		const deployments = await this.getDeployments(resourceGroup)
		const foundResource = deployments.find((res) => {
			const modelDetail = res.details?.resources?.backend_details?.model
			if (modelVersion) {
				return modelDetail?.name === modelName && modelDetail?.version === modelVersion
			} else {
				return modelDetail?.name === modelName
			}
		})
		if (foundResource) {
			const apiUrl = getLLMServiceUrl()
			return `${apiUrl}/inference/deployments/${foundResource.id}`
		} else {
			// fallback to the first matched model without version
			const defaultDeployment = deployments.find(
				(res) => res.details?.resources?.backend_details?.model?.name === modelName,
			)
			if (defaultDeployment) {
				const apiUrl = getLLMServiceUrl()
				return `${apiUrl}/inference/deployments/${defaultDeployment.id}`
			} else {
				Logger.log(`Cannot find the deployment URL for the model ${modelName}`)
				return ""
			}
		}
	}

	public async getCompletionUrl(modelName: string, modelVersion?: string, resourceGroup?: string): Promise<string> {
		const deploymentUrl = await this.getDeploymentUrl(modelName, modelVersion, resourceGroup)
		Logger.log(`Deployment URL: ${deploymentUrl}`)
		if (deploymentUrl) {
			const uriParameters = `chat/completions?api-version=2023-05-15`
			return `${deploymentUrl}/${uriParameters}`
		} else {
			return ""
		}
	}

	private getCompletionRequestConfig(resourceGroup?: string): any {
		return {
			headers: {
				"Content-Type": "application/json",
				"AI-Resource-Group": resourceGroup || "default",
			},
		}
	}

	public async requestCompletion(
		modelName: string,
		payload: any,
		modelVersion?: string,
		resourceGroup?: string,
	): Promise<Readable | undefined> {
		const url = await this.getCompletionUrl(modelName, modelVersion, resourceGroup)
		Logger.log(`Requesting completion from URL: ${url}`)
		Logger.log(`Payload: ${JSON.stringify(payload)}`)

		try {
			const response = await axios.post(url, payload, {
				...this.getCompletionRequestConfig(resourceGroup),
				responseType: "stream",
			})

			if (response.status === 200) {
				Logger.log(`Streaming response received.`)
				return response.data
			} else {
				Logger.log(`Unexpected status code: ${response.status}`)
			}
		} catch (error: any) {
			Logger.log(`Streaming request failed: ${error}`)
		}

		return undefined
	}
}
