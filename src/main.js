import './style.css';
import * as ort from 'onnxruntime-web';
import { OnnxModel } from './onnxModel';

async function getGPUInfo() {
    if (!navigator.gpu) {
        return 'WebGPU is not supported in this browser';
    }

    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            return 'WebGPU adapter not found';
        }

        // Get adapter info
        const info = await adapter.requestAdapterInfo();
        
        // Request device to get limits and features
        const device = await adapter.requestDevice();
        
        return `
            <div class="gpu-info">
                <h3>GPU Information:</h3>
                <ul>
                    <li>Vendor: ${info.vendor}</li>
                    <li>Architecture: ${info.architecture}</li>
                    <li>Description: ${info.description || 'Not available'}</li>
                    <li>Device: ${info.device || 'Not available'}</li>
                    <li>Max Buffer Size: ${device.limits.maxBufferSize / (1024 * 1024)} MB</li>
                    <li>Max Compute Invocations: ${device.limits.maxComputeInvocationsPerWorkgroup}</li>
                    <li>Max Storage Buffer Binding Size: ${device.limits.maxStorageBufferBindingSize / (1024 * 1024)} MB</li>
                </ul>
            </div>
        `;
    } catch (error) {
        console.error('Error getting GPU info:', error);
        return `WebGPU Error: ${error.message}`;
    }
}

async function initializeApp() {
    const container = document.querySelector('.container');

    // Add GPU info section with loading state
    const gpuInfo = document.createElement('div');
    gpuInfo.id = 'gpuInfo';
    gpuInfo.textContent = 'Loading GPU information...';
    container.insertBefore(gpuInfo, container.firstChild);

    // Update GPU info
    gpuInfo.innerHTML = await getGPUInfo();

    const modelUrl = '/models/model.onnx';
    const model = new OnnxModel(modelUrl);

    const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const dataB = Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
    const tensorA = new ort.Tensor('float32', dataA, [3, 4]);
    const tensorB = new ort.Tensor('float32', dataB, [4, 3]);
    const feeds = { a: tensorA, b: tensorB };

    const inferenceButton = document.getElementById('inferenceButton');
    const outputElement = document.getElementById('output');

    inferenceButton.addEventListener('click', async () => {
        try {
            inferenceButton.disabled = true;
            const output = await model.runInference(feeds);
            
            // Detailed output information
            const outputInfo = Object.entries(output).map(([key, tensor]) => ({
                name: key,
                shape: tensor.dims,
                data: Array.from(tensor.data).slice(0, 10),
                type: tensor.type,
                provider: model.session.handler._ep?.name || 'unknown'
            }));
            
            outputElement.innerHTML = `
                <h3>Inference Results:</h3>
                <pre>${JSON.stringify(outputInfo, null, 2)}</pre>
            `;
        } catch (err) {
            outputElement.innerHTML = `
                <div class="error">
                    <h3>Inference Error:</h3>
                    <pre>${err.message}</pre>
                    <p>Stack: ${err.stack}</p>
                </div>
            `;
        } finally {
            inferenceButton.disabled = false;
        }
    });
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', initializeApp);