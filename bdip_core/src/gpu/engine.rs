use crate::error::BdipError;

pub struct GpuEngine {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuEngine {
    pub fn new() -> Result<Self, BdipError> {
        let instance = wgpu::Instance::default();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| BdipError::Gpu(format!("No suitable GPU adapter found: {}", e)))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Bdip Headless Device"),
            required_features: wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .map_err(|e| BdipError::Gpu(format!("Device request failed: {:?}", e)))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_headless_init() {
        let engine = GpuEngine::new();
        assert!(
            engine.is_ok(),
            "Failed to initialize WebGPU engine headlessly"
        );
    }
}
