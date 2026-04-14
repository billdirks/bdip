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
            // Insert at the front so that redo_stack[0] is always the next item to
            // be re-applied by `redo()`, satisfying the `redo_transforms()` contract.
            self.redo_stack.insert(0, t);
            Some(())
        } else {
            None
        }
    }

    pub fn redo(&mut self) -> Option<()> {
        if !self.can_redo() {
            return None;
        }
        let t = self.redo_stack.remove(0);
        self.applied_transforms.push(t);
        Some(())
    }

    pub fn applied_transforms(&self) -> &[Transformation] {
        &self.applied_transforms
    }

    /// Returns `true` when there is at least one applied transform that can be undone.
    pub fn can_undo(&self) -> bool {
        !self.applied_transforms.is_empty()
    }

    /// Returns `true` when there is at least one undone transform that can be redone.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Returns the redo stack in redo order: index 0 is the next item that would be
    /// re-applied by a call to `redo()`.
    pub fn redo_transforms(&self) -> &[Transformation] {
        &self.redo_stack
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

    #[test]
    fn test_can_undo_empty() {
        let hm = HistoryManager::new();
        assert!(!hm.can_undo());
    }

    #[test]
    fn test_can_undo_after_apply() {
        let mut hm = HistoryManager::new();
        hm.apply(Transformation::Grayscale);
        assert!(hm.can_undo());
    }

    #[test]
    fn test_can_undo_false_after_all_undone() {
        let mut hm = HistoryManager::new();
        hm.apply(Transformation::Grayscale);
        hm.undo();
        assert!(!hm.can_undo());
    }

    #[test]
    fn test_can_redo_empty() {
        let hm = HistoryManager::new();
        assert!(!hm.can_redo());
    }

    #[test]
    fn test_can_redo_after_undo() {
        let mut hm = HistoryManager::new();
        hm.apply(Transformation::Invert);
        hm.undo();
        assert!(hm.can_redo());
    }

    #[test]
    fn test_can_redo_false_after_apply() {
        let mut hm = HistoryManager::new();
        hm.apply(Transformation::Invert);
        hm.undo();
        hm.apply(Transformation::Grayscale);
        assert!(!hm.can_redo());
    }

    #[test]
    fn test_redo_transforms_order() {
        let mut hm = HistoryManager::new();
        hm.apply(Transformation::Brightness(0.3));
        hm.apply(Transformation::Saturation(0.5));
        hm.undo(); // undoes S(0.5)
        hm.undo(); // undoes B(0.3) — last undone, so next to redo
        // redo_transforms()[0] must be the next item re-applied by redo(): B(0.3).
        let redo = hm.redo_transforms();
        assert_eq!(redo.len(), 2);
        assert_eq!(redo[0], Transformation::Brightness(0.3));
        assert_eq!(redo[1], Transformation::Saturation(0.5));
    }
}
