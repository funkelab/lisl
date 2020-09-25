import copy
import gunpowder as gp
import logging
import numpy as np

logger = logging.getLogger(__name__)


class RandomBranchSelector:

    def __init__(self, num_sources, probabilities=None, repetitions=1):
        '''Create random points in a provided ROI with the given density.

        Args:

            repetitions (int):

                How many times the generator will be used in a pipeline. Only
                the first request to the RandomSource will have a random source
                chosen. Future calls will use the same source.
        '''
        self.repetitions = repetitions
        self.num_sources = num_sources
        self.probabilities = probabilities

        # automatically normalize probabilities to sum to 1
        if self.probabilities is not None:
            self.probabilities = [
                float(x) / np.sum(probabilities)
                for x in self.probabilities
            ]

        if self.probabilities is not None:
            assert self.num_sources == len(
                self.probabilities), "if probabilities are specified, they " \
                                     "need to be given for each batch " \
                                     "provider added to the RandomProvider"
        self.iteration = 0

    def get_random_source(self):
        '''Get a randomly chosen source. If `repetitions` is larger than 1, the
        previously chosen source will be given.
        '''
        if self.iteration % self.repetitions == 0:

            self.choice = np.random.choice(list(range(self.num_sources)),
                                           p=self.probabilities)
        self.iteration += 1
        return self.choice


class RandomProvider(gp.BatchProvider):
    '''Randomly selects one of the upstream providers based on a
    RandomBranchSelector:: (a + b + c) + RandomProvider() will create a
    provider that randomly relays requests to providers ``a``, ``b``, or ``c``.
    Array and point keys of ``a``, ``b``, and ``c`` should be the same. The
    RandomBranchSelector determines which upstream branch to pick for each
    request.

    Args:

        random_branch_selector (:class:`RandomBranchSelector`)

            The random branch selector to determine which upstream branch to
            choose.
    '''

    def __init__(self, random_branch_selector):
        self.random_branch_selector = random_branch_selector

    def setup(self):

        assert len(self.get_upstream_providers()) > 0,\
            "at least one batch provider must be added to the RandomProvider"

        common_spec = None

        # advertise outputs only if all upstream providers have them
        for provider in self.get_upstream_providers():

            if common_spec is None:
                common_spec = copy.deepcopy(provider.spec)
            else:
                for key, spec in list(common_spec.items()):
                    if key not in provider.spec:
                        del common_spec[key]

        for key, spec in common_spec.items():
            self.provides(key, spec)

    def provide(self, request):

        source_idx = self.random_branch_selector.get_random_source()
        source = self.get_upstream_providers()[source_idx]
        logger.debug("Branch chosen: %s", source)
        return source.request_batch(request)
