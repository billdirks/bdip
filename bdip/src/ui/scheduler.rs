use bdip_core::Transformation;

/// Captures everything the background worker needs to replay a render.
#[derive(Debug, Clone)]
pub(super) enum RenderRequest {
    /// Standard preview or commit render — produces an iced image handle.
    Preview {
        render_list: Vec<Transformation>,
        width: u32,
        height: u32,
    },
    /// Full-resolution 16-bit render for saving.
    Save {
        render_list: Vec<Transformation>,
        width: u32,
        height: u32,
    },
}

/// Returned by [`RenderScheduler::request`].
#[derive(Debug, PartialEq)]
pub(super) enum ScheduleResult {
    /// No task is in-flight. The caller should dispatch this request and
    /// embed the provided generation counter in the task.
    Dispatch(u64),
    /// A task is already in-flight. The request has been queued for dispatch
    /// once the current task completes.
    Queued,
}

/// Returned by [`RenderScheduler::complete`].
#[derive(Debug)]
pub(super) enum CompleteResult {
    /// The generation matches the latest dispatch. Contains the pending
    /// request (if any) that should be dispatched next.
    Accept(Option<RenderRequest>),
    /// The generation is stale — a newer task was dispatched after this one.
    /// The caller should discard the result.
    Stale,
}

/// Two-state state machine (idle / in-flight) with a one-slot queue for the
/// most recent pending render request.
///
/// At most one GPU task is in-flight at a time. If a new request arrives while
/// one is in-flight, it replaces the previous pending slot so that only the
/// latest request is executed once the current task completes.
pub(super) struct RenderScheduler {
    is_rendering: bool,
    pending: Option<RenderRequest>,
    generation: u64,
}

impl RenderScheduler {
    pub(super) fn new() -> Self {
        RenderScheduler {
            is_rendering: false,
            pending: None,
            generation: 0,
        }
    }

    /// Submit a render request. If idle, transitions to in-flight and returns
    /// `Dispatch` with the new generation. If already in-flight, replaces any
    /// existing pending slot and returns `Queued`.
    pub(super) fn request(&mut self, req: RenderRequest) -> ScheduleResult {
        if self.is_rendering {
            // Only the latest request is retained; earlier pending values are
            // discarded because they would produce stale intermediate results.
            self.pending = Some(req);
            ScheduleResult::Queued
        } else {
            self.is_rendering = true;
            self.generation += 1;
            ScheduleResult::Dispatch(self.generation)
        }
    }

    /// Signal that a background task with the given generation has completed.
    ///
    /// If the generation matches the current generation, transitions to idle
    /// and returns `Accept` with any pending request. If the generation is
    /// older (stale), returns `Stale` without changing state, because the
    /// in-flight task may still complete in the future.
    pub(super) fn complete(&mut self, generation_id: u64) -> CompleteResult {
        if generation_id != self.generation {
            return CompleteResult::Stale;
        }
        self.is_rendering = false;
        CompleteResult::Accept(self.pending.take())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_preview_request() -> RenderRequest {
        RenderRequest::Preview {
            render_list: vec![],
            width: 100,
            height: 100,
        }
    }

    #[test]
    fn test_request_when_idle_returns_dispatch() {
        let mut sched = RenderScheduler::new();
        let result = sched.request(make_preview_request());
        assert_eq!(result, ScheduleResult::Dispatch(1));
    }

    #[test]
    fn test_request_when_in_flight_returns_queued() {
        let mut sched = RenderScheduler::new();
        sched.request(make_preview_request());
        let result = sched.request(make_preview_request());
        assert_eq!(result, ScheduleResult::Queued);
    }

    #[test]
    fn test_pending_replaces_previous_pending() {
        let mut sched = RenderScheduler::new();
        sched.request(make_preview_request());
        sched.request(RenderRequest::Preview {
            render_list: vec![Transformation::Brightness(0.3)],
            width: 100,
            height: 100,
        });
        // Third request should overwrite the second.
        sched.request(RenderRequest::Preview {
            render_list: vec![Transformation::Brightness(0.9)],
            width: 100,
            height: 100,
        });
        // Complete generation 1 and check pending carries the third request.
        let CompleteResult::Accept(Some(pending)) = sched.complete(1) else {
            panic!("expected Accept(Some(…))");
        };
        let RenderRequest::Preview { render_list, .. } = pending else {
            panic!("expected Preview variant");
        };
        assert_eq!(render_list, vec![Transformation::Brightness(0.9)]);
    }

    #[test]
    fn test_complete_matching_generation_returns_accept() {
        let mut sched = RenderScheduler::new();
        let ScheduleResult::Dispatch(g) = sched.request(make_preview_request()) else {
            panic!("expected Dispatch");
        };
        let result = sched.complete(g);
        assert!(matches!(result, CompleteResult::Accept(None)));
    }

    #[test]
    fn test_complete_returns_pending_request() {
        let mut sched = RenderScheduler::new();
        let ScheduleResult::Dispatch(g) = sched.request(make_preview_request()) else {
            panic!("expected Dispatch");
        };
        sched.request(make_preview_request());
        let result = sched.complete(g);
        assert!(matches!(result, CompleteResult::Accept(Some(_))));
    }

    #[test]
    fn test_complete_stale_generation_returns_stale() {
        let mut sched = RenderScheduler::new();
        sched.request(make_preview_request()); // gen 1 in-flight
        // Simulate completing an older, never-dispatched generation (0).
        let result = sched.complete(0);
        assert!(matches!(result, CompleteResult::Stale));
        // Scheduler must still consider itself in-flight.
        let result2 = sched.request(make_preview_request());
        assert_eq!(result2, ScheduleResult::Queued);
    }

    #[test]
    fn test_generation_increments_per_dispatch() {
        let mut sched = RenderScheduler::new();
        let ScheduleResult::Dispatch(g1) = sched.request(make_preview_request()) else {
            panic!("expected Dispatch");
        };
        sched.complete(g1);

        let ScheduleResult::Dispatch(g2) = sched.request(make_preview_request()) else {
            panic!("expected Dispatch");
        };
        sched.complete(g2);

        let ScheduleResult::Dispatch(g3) = sched.request(make_preview_request()) else {
            panic!("expected Dispatch");
        };

        assert!(g1 < g2);
        assert!(g2 < g3);
    }
}
