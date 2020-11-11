import pandas
import numpy
import cv2
from imgaug import augmenters as iaa
from matplotlib import pyplot


def pillar_adder(size=(512, 512), pillars=[
    (slice(50, 300), slice(50, 200)),
    (slice(250, 500), slice(300, 450))
], border_size=(50, 40)):
    """ Some of the yeast cell experiments contain pillars to keep the cells in
    place. This similates such pillars, by returning a function that will add
    pillars at the 2D slices in `pillars` of a `sz` x `sz` image. Returns
    a function f(image) -> image_with_pillars"""
    pillars_ = pillars
    pillars = numpy.zeros(size)

    r0 = 19
    r1 = 21
    br = 25

    for box in pillars_:
        a, b = box
        box = slice(a.start + br, a.stop - br), slice(b.start + br, b.stop - br)
        pillars[box] = 1

    pillars = cv2.dilate(
        pillars, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * br + 1, 2 * br + 1), (br, br)))

    pillars_inner = cv2.erode(
        pillars.copy(), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r0 // 2, r0 // 2), (r0 // 2 // 2, r0 // 2 // 2)))

    pillars_inner = pillars_inner - cv2.erode(
        pillars_inner, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r0, r0), (r0 // 2, r0 // 2)))

    pillars_outer = cv2.dilate(
        pillars, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r1 // 2, r1 // 2), (r1 // 2 // 2, r1 // 2 // 2)))

    pillars_outer = pillars_outer - cv2.erode(
        pillars_outer, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r1, r1), (r1 // 2, r1 // 2)))

    pillars_outer = cv2.GaussianBlur(pillars_outer, (21, 21), 2, 2)
    pillars_inner = cv2.GaussianBlur(pillars_inner, (21, 21), 2, 2)

    pillars_outer /= pillars_outer.max()
    pillars_inner /= pillars_inner.max()

    def add_pillars(image):
        with_pillars = 0.2 * image + 0.8 * numpy.maximum(image, pillars_inner)
        with_pillars = 0.2 * with_pillars + 0.8 * numpy.minimum(with_pillars, 1 - pillars_outer)
        return with_pillars

    return add_pillars


consistent_augmentations = iaa.Sequential([
    iaa.Fliplr(),
    iaa.Flipud(),
    iaa.Rot90(k=(0, 3)),
    iaa.GammaContrast(gamma=(0.9, 1.1)),
    #     iaa.SigmoidContrast(cutoff=0.5),
    iaa.LinearContrast(alpha=(0.95, 1.05))
], random_order=True)

seperate_augmentations = iaa.PiecewiseAffine((0.02, 0.03))


def random_cells(n, size=(512, 512),
                 min_distance_boundary=50,    # minimum distance of center from image boundary
                 r0_range=(2, 14),            # range of the first radius
                 r1_factor_range=(0.7, 1.3),  # range of the second radius as a factor of the first.
                 p_white_outside=1.0,         # chance the outside of a cell is white.
                 ):
    """Create a data frame of `n` random ellipses with one radius in `r0_range`, and
    the other radius is a factor from `r1_factor_range` of radius 0. The centers are
    at least `min_distance_boundary` from the border of size."""

    def randint_range(a, b, dtype=numpy.int):
        return (a + numpy.random.rand(n) * (b - a)).astype(dtype)

    d = min_distance_boundary
    r0 = randint_range(*r0_range)
    r1_factor = randint_range(*r1_factor_range, dtype=numpy.float)

    return pandas.DataFrame({
        'centerx': randint_range(d, size[0] - d),
        'centery': randint_range(d, size[1] - d),
        'radius0': r0, 'radius1': (r0 * r1_factor).astype(numpy.int),
        'angle':  randint_range(0, 360),
        'white-outside': numpy.random.rand(n) < p_white_outside
    })


def create_background(cores,
                      spatial_blur_std=1.5,
                      background_intensity=0.4,
                      background_contrast=0.00188,
                      core_contrast=0.0282,
                      ):
    """Creates a noisy, blurred background with different intensities
    for where there are cell and where there is nothing"""
    size = cores.shape
    n = (int(spatial_blur_std * 10) // 2) * 2 + 1
    background = cv2.GaussianBlur(numpy.random.randn(*size), (n, n), spatial_blur_std , spatial_blur_std )

    background /= background.std()
    cores = (cores > 0)
    a, b, z = background_contrast, core_contrast, background_intensity
    background = numpy.clip(z + (a + (b-a) * cores) * background, 0, 1)
    return background


def create_sample(size, cells,
                  spatial_blur_std=1.5,
                  background_intensity=0.4,
                  background_contrast=0.01,
                  core_contrast=0.15,
                  augmenter=seperate_augmentations,
                  ):
    """Create an image with cells as defined in `cells`"""
    cores = numpy.zeros(size)
    inner = numpy.zeros(size)
    outer = numpy.zeros(size)

    def draw_cell(x, y, r0, r1, angle, white_outside, label):
        nonlocal cores, inner, outer
        cores = cv2.ellipse(
            cores, (x, y), (r0, r1), angle,
            0, 360, label, -1
        )
        a, b = (inner, outer) if white_outside else (outer, inner)
        a = cv2.ellipse(a, (x, y), (r0 - 1, r1 - 1), angle, 0, 360, 1., -1)
        b = cv2.ellipse(b, (x, y), (r0 + 2, r1 + 2), angle, 0, 360, 1., -1)

    for label, (_, cell) in enumerate(cells.iterrows()):
        draw_cell(*cell[['centerx', 'centery', 'radius0', 'radius1', 'angle', 'white-outside']].values, label)

    aug = augmenter.to_deterministic()
    for im in [inner, outer, cores]:
        im[:] = aug.augment_images([im])[0]

    background = create_background(cores,
                                   spatial_blur_std=spatial_blur_std,
                                   background_intensity=background_intensity,
                                   background_contrast=background_contrast,
                                   core_contrast=core_contrast)

    for im in [inner, outer]:
        im[:] = im - cv2.erode(im, numpy.ones((3, 3)))
        im[:] = cv2.GaussianBlur(im, (11, 11), 2, 2)
        im[:] = im / im.max()

    cells = outer - inner
    cells -= cells.min(); cells /= cells.max()  # scale between 0 and 1
    return background + 0.5 * cells - 0.25, cores


def create_samples(n_images, n_cells_per_image=100,
                   size=(512, 512),
                   min_distance_boundary=50,    # minimum distance of center from image boundary
                   r0_range=(2, 14),            # range of the first radius
                   r1_factor_range=(0.7, 1.3),  # range of the second radius as a factor of the first.
                   spatial_blur_std=1.5,
                   background_intensity=0.4,
                   background_contrast=0.00188,
                   core_contrast=0.0752,
                   p_white_outside=1.0,
                  ):
    """Creates `n` `sz` x `sz` synthetic images of out of focus cells, 
    with m cells in each one. Then for each of the `n` images, `r` repetitions
    are made using image augmentation. Resulting in a `r` x `n` x `sz` x `sz`
    image array. Moreover, an array with the cell borders and cell centers
    are returned. The latter can be used as labels in the segmentation learning
    task"""
    add_pillars = pillar_adder(size)

    images = numpy.zeros((n_images, ) + size)
    labels = numpy.zeros((n_images, ) + size, dtype=numpy.int32)

    for image, label in zip(images, labels):
        cells = random_cells(n_cells_per_image, size=size,
                             min_distance_boundary=min_distance_boundary,
                             r0_range=r0_range, r1_factor_range=r1_factor_range,
                             p_white_outside=p_white_outside)
        image[:], label[:] = create_sample(
            size, cells,
            spatial_blur_std=spatial_blur_std,
            background_intensity=background_intensity,
            background_contrast=background_contrast,
            core_contrast=core_contrast
        )
        image[:] = add_pillars(image)

    # images -= images.min()
    # images /= images.max()
    images = (255 * images[..., None]).repeat(3, -1).astype(numpy.uint8)

    return images, labels


def colored_segmentation_map(labels, cmap='hsv', alpha=0.2, background = [0,0,0,0]):
    colors = pyplot.get_cmap(cmap)(numpy.linspace(0, 1, labels.max() + 1))
    colors[0] = background
    colors[1:, 3] = alpha
    return (colors[labels] * 255).astype(numpy.uint8)
