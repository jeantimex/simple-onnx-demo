import * as ort from 'onnxruntime-web/webgpu';

export class OnnxModel {
    constructor(modelUrl) {
        this.modelUrl = modelUrl;
        this.session = null;
        this.isLoading = true;
        this.error = null;
        this.usingWebGPU = false;

        // Set WASM paths
        ort.env.wasm.wasmPaths = {
            'ort-wasm.wasm': '/ort-wasm.wasm',
            'ort-wasm-simd.wasm': '/ort-wasm-simd.wasm',
            'ort-wasm-threaded.wasm': '/ort-wasm-threaded.wasm'
        };

        this.initialize();
    }

    async checkWebGPUSupport() {
        if (!navigator.gpu) {
            console.log('WebGPU is not supported in this browser');
            return false;
        }
        
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.log('No WebGPU adapter found');
                return false;
            }
            const device = await adapter.requestDevice();
            return !!device;
        } catch (error) {
            console.log('WebGPU device creation failed:', error);
            return false;
        }
    }

    async initialize() {
        try {
            // Try WASM first
            try {
                console.log('Attempting WASM initialization...');
                const wasmOptions = {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all',
                    enableCpuMemArena: true
                };
                
                this.session = await ort.InferenceSession.create(this.modelUrl, wasmOptions);
                console.log('WASM initialization successful');
                
                // Now try WebGPU if available
                if (await this.checkWebGPUSupport()) {
                    console.log('WebGPU support detected, attempting WebGPU initialization...');
                    try {
                        const webGPUOptions = {
                            executionProviders: ['webgpu'],
                            graphOptimizationLevel: 'all',
                            enableCpuMemArena: true,
                            webgpuFlags: {
                                preferWebGPU: true,
                                enableFloat16: true
                            }
                        };
                        
                        const webGPUSession = await ort.InferenceSession.create(this.modelUrl, webGPUOptions);
                        this.session = webGPUSession;
                        this.usingWebGPU = true;
                        console.log('WebGPU initialization successful');
                    } catch (webgpuError) {
                        console.log('WebGPU initialization failed, keeping WASM session:', webgpuError);
                        // Keep using the WASM session we already created
                    }
                }
            } catch (wasmError) {
                console.error('WASM initialization failed:', wasmError);
                throw wasmError;
            }

            console.log('Final execution provider:', this.session.handler?._ep?.name);
            this.isLoading = false;
            this.updateUI();

        } catch (err) {
            console.error('Initialization error:', err);
            this.error = err;
            this.isLoading = false;
            this.updateUI();
        }
    }

    async runInference(feeds) {
        if (!this.session) {
            throw new Error('Model is not loaded yet');
        }
        try {
            // Run inference
            console.log('Running inference with feeds:', feeds);
            const results = await this.session.run(feeds);
            
            // Log results
            console.log('Inference complete. Results:', results);
            
            return results;
        } catch (error) {
            console.error('Inference error:', error);
            throw error;
        }
    }

    updateUI() {
        const statusElement = document.getElementById('modelStatus');
        const inferenceButton = document.getElementById('inferenceButton');

        if (this.isLoading) {
            statusElement.textContent = 'Loading model...';
            inferenceButton.disabled = true;
        } else if (this.error) {
            statusElement.textContent = `Error loading model: ${this.error.message}`;
            statusElement.classList.add('error');
            inferenceButton.disabled = true;
        } else {
            const provider = this.usingWebGPU ? 'WebGPU' : 'WASM';
            statusElement.textContent = `Model loaded successfully (using ${provider})`;
            inferenceButton.disabled = false;
        }
    }
}