from gunpowder.profiling import Timing
import gunpowder as gp
import logging

logger = logging.getLogger(__name__)


class RejectArray(gp.BatchFilter):
    """
        Rejects array if it's size is 0.
    """

    def __init__(self, ensure_nonempty):
        self.ensure_nonempty = ensure_nonempty

    def setup(self):
        self.upstream_provider = self.get_upstream_provider()

    def provide(self, request):

        report_next_timeout = 2
        num_rejected = 0

        timing = Timing(self)
        timing.start()

        have_good_batch = False
        while not have_good_batch:

            batch = self.upstream_provider.request_batch(request)

            if batch.arrays[self.ensure_nonempty].data.size != 0:

                have_good_batch = True
                logger.debug(
                    "Accepted batch with shape: %s",
                    batch.arrays[self.ensure_nonempty].data.shape)

            else:

                num_rejected += 1

                if timing.elapsed() > report_next_timeout:
                    logger.info(
                        "rejected %s batches, been waiting for a good one "
                        "since %s",
                        num_rejected,
                        report_next_timeout)
                    report_next_timeout *= 2

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
