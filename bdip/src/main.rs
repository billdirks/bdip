use anyhow::Result;
use bdip_core::{BdipError, HistoryManager, Transformation};

fn main() -> Result<()> {
    println!("bdip v0.1.0");
    
    // Quick compile-time check to ensure imports work
    let mut _hm = HistoryManager::new();
    let _t = Transformation::Grayscale;
    let _err = BdipError::UnsupportedFormat("test".into());

    Ok(())
}
