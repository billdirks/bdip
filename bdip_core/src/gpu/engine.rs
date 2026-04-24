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

        let mut features = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
        if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Bdip Headless Device"),
            required_features: features,
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

    pub fn supports_timestamps(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
    }

    pub fn timestamp_period_ns(&self) -> f32 {
        self.queue.get_timestamp_period()
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
