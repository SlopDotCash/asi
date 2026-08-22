# Native supervised continual-learning suite

Issue #1578 pins Avalanche at
`ContinualAI/avalanche@eb075be393e1f458b2c352514ff6c17b5a2c0f4e` (MIT; audited
18 August 2026) and its paper at arXiv `2302.01766`. Dataset authorities are the original MNIST
site (60,000 train and 10,000 test 28×28 images) and the University of Toronto CIFAR-100 release
(100 fine classes, 600 images each). ASI's existing IPMNIST anchor remains Elsayed and Mahmood,
ICLR 2024, arXiv `2404.00781`, and its separately audited UPGD code revision; this suite does not
replace the full `upgd_ipmnist` runner.

Every native catalog and result also binds the complete prospective Avalanche
qualification plan by SHA-256
`ee85d404886ec2ae3412f9bec888e36cb1a984b41185af90cd8d1e36f7975053`.
That plan owns the audited source tree/archive/license, hash-locked runtime,
compatibility deviations, and ten unresolved execution blockers. The external
runtime test recomputes this digest from the plan bytes, so those inputs cannot
drift independently of native receipts. This binding does not authorize an
image build, dataset download, scenario execution, or parity claim.

The additive `asi.native_supervised_cl_development.v1` runner accepts caller-supplied exact
float32 images and int32 labels; it never downloads or writes data. It deterministically constructs
Split MNIST (five ascending class pairs), Rotated MNIST (fixed 0/45/90/135/180 degree rotations),
Split CIFAR-100 (twenty ascending five-class experiences), and IPMNIST (seeded per-task pixel
permutations). Four consumed development seeds bind example order and transformations. Task IDs and
boundaries are never passed to learners. Every arm sees identical task arrays in identical order.

The matched controls are online multinomial SGD, bounded replay SGD, an online running-centroid
classifier, and a literal frozen/no-learning reduction. The SGD arms have the same optimizer-call
budget: replay uses a retained example while its control repeats the current example. Predictions
precede updates. Receipts count
data examples and bytes, model queries, parameter updates, replay inserts/samples/peak bytes,
persistent numeric bytes, exact logical calls, and telemetry-only elapsed nanoseconds. Results are
permanently nonpromoting and negative outcomes must be retained. `asi-native-supervised-catalog
--catalog` reports setup metadata only; dataset execution requires explicit arrays through the API.

This qualification is not Avalanche parity or a scientific benchmark. Avalanche permits shuffled
class orders and task labels; RotatedMNIST uses torchvision/PIL `RandomRotation`, whereas the native
lane freezes an explicit NumPy nearest-neighbor transform. Avalanche Split CIFAR-100 defaults to
random crop and horizontal flip, which this deterministic lane excludes. The bounded runner uses a
linear model, one online pass, small development slices, no held-out evaluation matrix, and no deep
external strategies. IPMNIST qualification slices do not match its 200×5,000 full campaign budget.

Before comparison: pin dataset byte checksums and train/test split loaders; reproduce Avalanche task
membership, pixel transforms, normalization and augmentation at the pinned revision; freeze whether
task IDs/boundaries, replay, pretrained features, multi-epoch training, and dynamic heads are allowed;
add held-out after-task accuracy/forgetting/forward-transfer matrices; port strong deep replay and
regularization baselines; qualify accelerator timing and compute accounting; rerun the full existing
IPMNIST controls; and preregister untouched scientific seeds. No result or SOTA claim exists here.
