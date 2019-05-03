import numpy as np
from scarlet.psf import gaussian

import lsst.log
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.afw.geom.ellipses as afwEll
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
import lsst.afw.table as afwTable

from .source import LsstSource, LsstHistory
from .blend import LsstBlend
from .observation import LsstScene, LsstObservation


logger = lsst.log.Log.getLogger("meas.deblender.deblend")


def _getPsfFwhm(psf):
    """Calculate the FWHM of the `psf`
    """
    return psf.computeShape().getDeterminantRadius() * 2.35


def _estimateStdDev(exposure, statsMask):
    """Estimate the standard dev. of an image

    Take the median standard deviation of the `exposure`.
    """
    mi = exposure.getMaskedImage()
    statsCtrl = afwMath.StatisticsControl()
    statsCtrl.setAndMask(mi.getMask().getPlaneBitMask(statsMask))
    stats = afwMath.makeStatistics(mi.variance, mi.mask, afwMath.MEDIAN, statsCtrl)
    sigma = np.sqrt(stats.getValue(afwMath.MEDIAN))
    return sigma


def _getTargetPsf(shape, sigma=1/np.sqrt(2)):
    X = np.arange(shape[2])
    Y = np.arange(shape[1])
    X, Y = np.meshgrid(X, Y)
    coords = np.stack([Y, X])
    y0, x0 = (shape[1]-1) // 2, (shape[2]-1) // 2
    target_psf = gaussian(coords, y0, x0, 1, sigma)
    target_psf /= target_psf.sum()
    return target_psf


def deblend(mExposure, footprint, log, config):
    # Extract coordinates from each MultiColorPeak
    bbox = footprint.getBBox()
    xmin = bbox.getMinX()
    ymin = bbox.getMinY()
    peaks = [[pk.Iy-ymin, pk.Ix-xmin] for pk in footprint.peaks]

    # Create the data array from the masked images
    images = mExposure.image[:, bbox].array

    # Use the inverse variance as the weights
    if config.useWeights:
        weights = 1/mExposure.variance[:, bbox].array
    else:
        weights = np.ones_like(images)

    # Use the mask plane to mask bad pixels and
    # the footprint to mask out pixels outside the footprint
    if config.badMask is None:
        badMask = ["BAD", "CR", "NO_DATA", "SAT", "SUSPECT"]
    fpMask = afwImage.Mask(bbox)
    footprint.spans.setMask(fpMask, 1)
    fpMask = ~fpMask.getArray().astype(bool)
    badPixels = mExposure.getMask().getPlaneBitMask(badMask)
    mask = (mExposure.getMask()[bbox].array & badPixels) | fpMask[None, :]
    weights[mask > 0] = 0

    psfs = mExposure.computePsfImage(footprint.getCentroid()).array
    target_psf = _getTargetPsf(psfs.shape)

    observation = LsstObservation(images, psfs)
    scene = LsstScene(images.shape, psf=target_psf)
    bg_rms = np.array([_estimateStdDev(exposure, config.statsMask) for exposure in mExposure[bbox]])
    if config.storeHistory:
        Source = LsstHistory
    else:
        Source = LsstSource
    sources = [
        Source(center.astype(int), scene, observation, bg_rms,
               config.symmetric, config.monotonic, config.recenterPeriod)
        for center in peaks
    ]

    blend = LsstBlend(scene, sources, observation)
    blend.fit(config.maxIter, config.relativeError, False)

    return blend


class ScarletDeblendConfig(pexConfig.Config):
    """MultibandDeblendConfig

    Configuration for the multiband deblender.
    The parameters are organized by the parameter types, which are
    - Stopping Criteria: Used to determine if the fit has converged
    - Position Fitting Criteria: Used to fit the positions of the peaks
    - Constraints: Used to apply constraints to the peaks and their components
    - Other: Parameters that don't fit into the above categories
    """
    # Stopping Criteria
    maxIter = pexConfig.Field(dtype=int, default=200,
                              doc=("Maximum number of iterations to deblend a single parent"))
    relativeError = pexConfig.Field(dtype=float, default=1e-2,
                                    doc=("Relative error to use when determining stopping criteria"))

    # Blend Configuration options
    recenterPeriod = pexConfig.Field(dtype=int, default=5,
                                     doc=("Number of iterations between recentering"))
    exactLipschitz = pexConfig.Field(dtype=bool, default=True,
                                     doc=("Calculate exact Lipschitz constant in every step"
                                          "(True) or only calculate the approximate"
                                          "Lipschitz constant with significant changes in A,S"
                                          "(False)"))

    # Constraints
    sparse = pexConfig.Field(dtype=bool, default=True, doc="Make models compact and sparse")
    monotonic = pexConfig.Field(dtype=bool, default=True, doc="Make models monotonic")
    symmetric = pexConfig.Field(dtype=bool, default=False, doc="Make models symmetric")
    symmetryThresh = pexConfig.Field(dtype=float, default=1.0,
                                     doc=("Strictness of symmetry, from"
                                          "0 (no symmetry enforced) to"
                                          "1 (perfect symmetry required)."
                                          "If 'S' is not in `constraints`, this argument is ignored"))

    # Other scarlet paremeters
    useWeights = pexConfig.Field(dtype=bool, default=False, doc="Use inverse variance as deblender weights")
    usePsfConvolution = pexConfig.Field(
        dtype=bool, default=True,
        doc=("Whether or not to convolve the morphology with the"
             "PSF in each band or use the same morphology in all bands"))
    saveTemplates = pexConfig.Field(
        dtype=bool, default=True,
        doc="Whether or not to save the SEDs and templates")
    processSingles = pexConfig.Field(
        dtype=bool, default=False,
        doc="Whether or not to process isolated sources in the deblender")

    # Mask-plane restrictions
    badMask = pexConfig.Field(
        dtype=str, default=["BAD", "CR", "NO_DATA", "SAT,SUSPECT"],
        doc="Whether or not to process isolated sources in the deblender")
    statsMask = pexConfig.Field(dtype=str, default=["SAT", "INTRP", "NO_DATA"],
                                doc="Mask planes to ignore when performing statistics")
    maskLimits = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        default={},
        doc=("Mask planes with the corresponding limit on the fraction of masked pixels. "
             "Sources violating this limit will not be deblended."),
    )

    # Size restrictions
    maxNumberOfPeaks = pexConfig.Field(
        dtype=int, default=0,
        doc=("Only deblend the brightest maxNumberOfPeaks peaks in the parent"
             " (<= 0: unlimited)"))
    maxFootprintArea = pexConfig.Field(
        dtype=int, default=1000000,
        doc=("Maximum area for footprints before they are ignored as large; "
             "non-positive means no threshold applied"))
    maxFootprintSize = pexConfig.Field(
        dtype=int, default=0,
        doc=("Maximum linear dimension for footprints before they are ignored "
             "as large; non-positive means no threshold applied"))
    minFootprintAxisRatio = pexConfig.Field(
        dtype=float, default=0.0,
        doc=("Minimum axis ratio for footprints before they are ignored "
             "as large; non-positive means no threshold applied"))

    # Failure modes
    notDeblendedMask = pexConfig.Field(
        dtype=str, default="NOT_DEBLENDED", optional=True,
        doc="Mask name for footprints not deblended, or None")
    catchFailures = pexConfig.Field(
        dtype=bool, default=False,
        doc=("If True, catch exceptions thrown by the deblender, log them, "
             "and set a flag on the parent, instead of letting them propagate up"))
    propagateAllPeaks = pexConfig.Field(dtype=bool, default=False,
                                        doc=('Guarantee that all peaks produce a child source.'))


class ScarletDeblendTask(pipeBase.Task):
    """ScarletDeblendTask

    Split blended sources into individual sources.

    This task has no return value; it only modifies the SourceCatalog in-place.
    """
    ConfigClass = ScarletDeblendConfig
    _DefaultName = "scarletDeblend"

    def __init__(self, schema, peakSchema=None, **kwargs):
        """Create the task, adding necessary fields to the given schema.

        Parameters
        ----------
        schema: `lsst.afw.table.schema.schema.Schema`
            Schema object for measurement fields; will be modified in-place.
        peakSchema: `lsst.afw.table.schema.schema.Schema`
            Schema of Footprint Peaks that will be passed to the deblender.
            Any fields beyond the PeakTable minimal schema will be transferred
            to the main source Schema.  If None, no fields will be transferred
            from the Peaks.
        filters: list of str
            Names of the filters used for the eposures. This is needed to store the SED as a field
        **kwargs
            Passed to Task.__init__.
        """
        pipeBase.Task.__init__(self, **kwargs)
        if not self.config.conserveFlux and not self.config.saveTemplates:
            raise ValueError("Either `conserveFlux` or `saveTemplates` must be True")

        peakMinimalSchema = afwDet.PeakTable.makeMinimalSchema()
        if peakSchema is None:
            # In this case, the peakSchemaMapper will transfer nothing, but we'll still have one
            # to simplify downstream code
            self.peakSchemaMapper = afwTable.SchemaMapper(peakMinimalSchema, schema)
        else:
            self.peakSchemaMapper = afwTable.SchemaMapper(peakSchema, schema)
            for item in peakSchema:
                if item.key not in peakMinimalSchema:
                    self.peakSchemaMapper.addMapping(item.key, item.field)
                    # Because SchemaMapper makes a copy of the output schema you give its ctor, it isn't
                    # updating this Schema in place.  That's probably a design flaw, but in the meantime,
                    # we'll keep that schema in sync with the peakSchemaMapper.getOutputSchema() manually,
                    # by adding the same fields to both.
                    schema.addField(item.field)
            assert schema == self.peakSchemaMapper.getOutputSchema(), "Logic bug mapping schemas"
        self._addSchemaKeys(schema)
        self.schema = schema

    def _addSchemaKeys(self, schema):
        """Add deblender specific keys to the schema
        """
        self.runtimeKey = schema.addField('runtime', type=np.float32, doc='runtime in ms')

        self.nChildKey = schema.addField('deblend_nChild', type=np.int32,
                                         doc='Number of children this object has (defaults to 0)')
        self.psfKey = schema.addField('deblend_deblendedAsPsf', type='Flag',
                                      doc='Deblender thought this source looked like a PSF')
        self.tooManyPeaksKey = schema.addField('deblend_tooManyPeaks', type='Flag',
                                               doc='Source had too many peaks; '
                                               'only the brightest were included')
        self.tooBigKey = schema.addField('deblend_parentTooBig', type='Flag',
                                         doc='Parent footprint covered too many pixels')
        self.maskedKey = schema.addField('deblend_masked', type='Flag',
                                         doc='Parent footprint was predominantly masked')
        self.convergenceFailed = schema.addField('deblend_convergenceFailed', type='Flag',
                                                 doc='scarlet optimization did not converge before'
                                                     'config.maxIter')
        self.deblendFailedKey = schema.addField('deblend_failed', type='Flag',
                                                doc="Deblending failed on source")

        self.deblendSkippedKey = schema.addField('deblend_skipped', type='Flag',
                                                 doc="Deblender skipped this source")
        # self.log.trace('Added keys to schema: %s', ", ".join(str(x) for x in (
        #               self.nChildKey, self.tooManyPeaksKey, self.tooBigKey)))

    @pipeBase.timeMethod
    def run(self, mExposure, mergedSources):
        """Get the psf from each exposure and then run deblend().

        Parameters
        ----------
        mExposure: `MultibandExposure`
            The exposures should be co-added images of the same
            shape and region of the sky.
        mergedSources: `SourceCatalog`
            The merged `SourceCatalog` that contains parent footprints
            to (potentially) deblend.

        Returns
        -------
        fluxCatalogs: dict or None
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s.
            These are the flux-conserved catalogs with heavy footprints with
            the image data weighted by the multiband templates.
            If `self.config.conserveFlux` is `False`, then this item will be None
        templateCatalogs: dict or None
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s.
            These are catalogs with heavy footprints that are the templates
            created by the multiband templates.
            If `self.config.saveTemplates` is `False`, then this item will be None
        """
        return self.deblend(mExposure, mergedSources)

    @pipeBase.timeMethod
    def deblend(self, mExposure, sources):
        """Deblend a data cube of multiband images

        Parameters
        ----------
        mExposure: `MultibandExposure`
            The exposures should be co-added images of the same
            shape and region of the sky.
        sources: `SourceCatalog`
            The merged `SourceCatalog` that contains parent footprints
            to (potentially) deblend.

        Returns
        -------
        fluxCatalogs: dict or None
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s.
            These are the flux-conserved catalogs with heavy footprints with
            the image data weighted by the multiband templates.
            If `self.config.conserveFlux` is `False`, then this item will be None
        templateCatalogs: dict or None
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s.
            These are catalogs with heavy footprints that are the templates
            created by the multiband templates.
            If `self.config.saveTemplates` is `False`, then this item will be None
        """
        import time

        filters = mExposure.filters
        self.log.info("Deblending {0} sources in {1} exposure bands".format(len(sources), len(mExposure)))

        # Create the output catalogs
        templateCatalogs = {}
        for f in filters:
            _catalog = afwTable.SourceCatalog(sources.table.clone())
            _catalog.extend(sources)
            templateCatalogs[f] = _catalog

        n0 = len(sources)
        nparents = 0
        for pk, src in enumerate(sources):
            foot = src.getFootprint()
            bbox = foot.getBBox()
            logger.info("id: {0}".format(src["id"]))
            peaks = foot.getPeaks()

            # Since we use the first peak for the parent object, we should propagate its flags
            # to the parent source.
            src.assign(peaks[0], self.peakSchemaMapper)

            # Block of Skipping conditions
            if len(peaks) < 2 and not self.config.processSingles:
                for f in filters:
                    templateCatalogs[f][pk].set(self.runtimeKey, 0)
                continue
            if self._isLargeFootprint(foot):
                src.set(self.tooBigKey, True)
                self._skipParent(src, mExposure.mask)
                self.log.trace('Parent %i: skipping large footprint', int(src.getId()))
                continue
            if self._isMasked(foot, mExposure):
                src.set(self.maskedKey, True)
                mask = np.bitwise_or.reduce(mExposure.mask[:, bbox].array, axis=0)
                mask = afwImage.MaskX(mask, xy0=bbox.getMin())
                self._skipParent(src, mask)
                self.log.trace('Parent %i: skipping masked footprint', int(src.getId()))
                continue
            if len(peaks) > self.config.maxNumberOfPeaks:
                src.set(self.tooManyPeaksKey, True)
                msg = 'Parent {0}: Too many peaks, using the first {1} peaks'
                self.log.trace(msg.format(int(src.getId()), self.config.maxNumberOfPeaks))

            nparents += 1
            self.log.trace('Parent %i: deblending %i peaks', int(src.getId()), len(peaks))
            # Run the deblender
            try:
                t0 = time.time()
                # Build the parameter lists with the same ordering
                blend = deblend(mExposure, foot, self.log, self.config)
                tf = time.time()
                runtime = (tf-t0)*1000
                src.set(self.deblendFailedKey, False)
                src.set(self.runtimeKey, runtime)
            except Exception as e:
                if self.config.catchFailures:
                    self.log.warn("Unable to deblend source %d: %s" % (src.getId(), e))
                    src.set(self.deblendFailedKey, True)
                    src.set(self.runtimeKey, 0)
                    import traceback
                    traceback.print_exc()
                    continue
                else:
                    raise

            # Add the merged source as a parent in the catalog for each band
            templateParents = {}
            parentId = src.getId()
            for f in filters:
                templateParents[f] = templateCatalogs[f][pk]
                templateParents[f].set(self.runtimeKey, runtime)

            # Add each source to the catalogs in each band
            templateSpans = {f: afwGeom.SpanSet() for f in filters}
            nchild = 0
            for k, source in blend.sources:
                py, px = source.pixel_center
                if source.morph.sum() == 0 or source.sed.sum() == 0:
                    src.set(self.deblendSkippedKey, True)
                    if not self.config.propagateAllPeaks:
                        # We don't care
                        continue
                    # We need to preserve the peak: make sure we have enough info to create a minimal
                    # child src
                    msg = "Peak at {0} failed deblending.  Using minimal default info for child."
                    self.log.trace(msg.format(px, py))
                else:
                    src.set(self.deblendSkippedKey, False)
                models = source.modelToHeavy(xy0=bbox.getMin())
                # TODO: We should eventually write the morphology and SED to the catalog
                # morph = source.morphToHeavy(xy0=bbox.getMin())
                # sed = source.sed / source.sed.sum()

                for f in filters:
                    if len(models[f].getPeaks()) != 1:
                        err = "Heavy footprint should have a single peak, got {0}"
                        raise ValueError(err.format(len(models[f].peaks)))
                    cat = templateCatalogs[f]
                    child = self._addChild(parentId, cat, models[f])
                    if parentId == 0:
                        child.setId(src.getId())
                        child.set(self.runtimeKey, runtime)
                    else:
                        templateSpans[f] = templateSpans[f].union(models[f].getSpans())
                nchild += 1

            # Child footprints may extend beyond the full extent of their parent's which
            # results in a failure of the replace-by-noise code to reinstate these pixels
            # to their original values.  The following updates the parent footprint
            # in-place to ensure it contains the full union of itself and all of its
            # children's footprints.
            for f in filters:
                templateParents[f].set(self.nChildKey, nchild)
                templateParents[f].getFootprint().setSpans(templateSpans[f])

        K = len(list(templateCatalogs.values())[0])
        self.log.info('Deblended: of %i sources, %i were deblended, creating %i children, total %i sources'
                      % (n0, nparents, K-n0, K))
        return templateCatalogs

    def _isLargeFootprint(self, footprint):
        """Returns whether a Footprint is large

        'Large' is defined by thresholds on the area, size and axis ratio.
        These may be disabled independently by configuring them to be non-positive.

        This is principally intended to get rid of satellite streaks, which the
        deblender or other downstream processing can have trouble dealing with
        (e.g., multiple large HeavyFootprints can chew up memory).
        """
        if self.config.maxFootprintArea > 0 and footprint.getArea() > self.config.maxFootprintArea:
            return True
        if self.config.maxFootprintSize > 0:
            bbox = footprint.getBBox()
            if max(bbox.getWidth(), bbox.getHeight()) > self.config.maxFootprintSize:
                return True
        if self.config.minFootprintAxisRatio > 0:
            axes = afwEll.Axes(footprint.getShape())
            if axes.getB() < self.config.minFootprintAxisRatio*axes.getA():
                return True
        return False

    def _isMasked(self, footprint, mExposure):
        """Returns whether the footprint violates the mask limits"""
        bbox = footprint.getBBox()
        mask = np.bitwise_or.reduce(mExposure.mask[:, bbox].array, axis=0)
        size = float(footprint.getArea())
        for maskName, limit in self.config.maskLimits.items():
            maskVal = mExposure.mask.getPlaneBitMask(maskName)
            _mask = afwImage.MaskX(mask & maskVal, xy0=bbox.getMin())
            unmaskedSpan = footprint.spans.intersectNot(_mask)  # spanset of unmasked pixels
            if (size - unmaskedSpan.getArea())/size > limit:
                return True
        return False

    def _skipParent(self, source, masks):
        """Indicate that the parent source is not being deblended

        We set the appropriate flags and masks for each exposure.

        Parameters
        ----------
        source: `lsst.afw.table.source.source.SourceRecord`
            The source to flag as skipped
        masks: list of `lsst.afw.image.MaskX`
            The mask in each band to update with the non-detection
        """
        fp = source.getFootprint()
        source.set(self.deblendSkippedKey, True)
        source.set(self.nChildKey, len(fp.getPeaks()))  # It would have this many if we deblended them all
        if self.config.notDeblendedMask:
            for mask in masks:
                mask.addMaskPlane(self.config.notDeblendedMask)
                fp.spans.setMask(mask, mask.getPlaneBitMask(self.config.notDeblendedMask))

    def _addChild(self, parentId, sources, heavy):
        """Add a child to a catalog

        This creates a new child in the source catalog,
        assigning it a parent id, adding a footprint,
        and setting all appropriate flags based on the
        deblender result.
        """
        assert len(heavy.getPeaks()) == 1
        src = sources.addNew()
        src.assign(heavy.getPeaks()[0], self.peakSchemaMapper)
        src.setParent(parentId)
        src.setFootprint(heavy)
        src.set(self.psfKey, False)
        src.set(self.runtimeKey, 0)
        return src