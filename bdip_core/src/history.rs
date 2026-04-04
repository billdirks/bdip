use crate::transformation::Transformation;

pub struct HistoryManager {
    applied_transforms: Vec<Transformation>,
    redo_stack: Vec<Transformation>,
}

impl Default for HistoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl HistoryManager {
    pub fn new() -> Self {
        Self {
            applied_transforms: Vec::new(),
            redo_stack: Vec::new(),
        }
    }

    pub fn apply(&mut self, t: Transformation) {
        self.applied_transforms.push(t);
        self.redo_stack.clear();
    }

    pub fn undo(&mut self) -> Option<()> {
        if let Some(t) = self.applied_transforms.pop() {
            self.redo_stack.push(t);
            Some(())
        } else {
            None
        }
    }

    pub fn redo(&mut self) -> Option<()> {
        if let Some(t) = self.redo_stack.pop() {
            self.applied_transforms.push(t);
            Some(())
        } else {
            None
        }
    }

    pub fn applied_transforms(&self) -> &[Transformation] {
        &self.applied_transforms
    }

    pub fn clear(&mut self) {
        self.applied_transforms.clear();
        self.redo_stack.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_initial_state() {
        let hm = HistoryManager::new();
        assert_eq!(hm.applied_transforms().len(), 0);
    }

    #[test]
    fn test_apply_and_clear() {
        let mut hm = HistoryManager::new();
        hm.apply(Transformation::Grayscale);
        assert_eq!(hm.applied_transforms().len(), 1);
        hm.clear();
        assert_eq!(hm.applied_transforms().len(), 0);
    }

    #[test]
    fn test_undo_redo() {
        let mut hm = HistoryManager::new();
        hm.apply(Transformation::Grayscale);
        hm.apply(Transformation::Invert);
        
        assert_eq!(hm.undo(), Some(()));
        assert_eq!(hm.applied_transforms().len(), 1);
        
        assert_eq!(hm.redo(), Some(()));
        assert_eq!(hm.applied_transforms().len(), 2);
    }

    #[test]
    fn test_apply_clears_redo() {
        let mut hm = HistoryManager::new();
        hm.apply(Transformation::Grayscale);
        hm.undo();
        hm.apply(Transformation::Contrast(0.8));
        assert_eq!(hm.redo(), None);
    }
    
    #[test]
    fn test_undo_empty_stack() {
        let mut hm = HistoryManager::new();
        assert_eq!(hm.undo(), None);
        assert_eq!(hm.redo(), None);
    }
}
