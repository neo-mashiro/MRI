using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace NiftiProcessor {
    
    public static class OtsuThreshold {
        /// <summary>
        /// Compute the histogram of the 3D volumetric brain image.
        /// </summary>
        private static List<int> ComputeHistogram(IReadOnlyList<dynamic> imageArray) {
            var histogram = new List<int>(new int[(int) imageArray.Max() + 1]);

            for (var i = 0; i < imageArray.Count; i++) {
                histogram[(int) imageArray[i]] += 1;
            }

            return histogram;
        }

        /// <summary>
        /// Compute the intensity threshold that segments the gray matter from the brain.
        /// </summary>
        public static int SegmentBrain(IReadOnlyList<dynamic> imageArray) {
            var histogram = ComputeHistogram(imageArray);
            var totalCount = histogram.Sum();

            // exhaustively searches for the threshold that minimizes the intra-class variance
            // for our two classes, this is equivalent to maximizing the inter-class variance
            var threshold = 0;
            var maxInterClassVariance = 0.0f;
            var maxVoxelIntensity = (int) imageArray.Max();

            for (var t = 1; t <= maxVoxelIntensity; t++) {
                var class0 = new int[t];
                var class1 = new int[maxVoxelIntensity + 1 - t];

                histogram.CopyTo(0, class0, 0, t);
                histogram.CopyTo(t, class1, 0, maxVoxelIntensity + 1 - t);

                var w0 = (float) class0.Sum() / totalCount;
                var w1 = (float) class1.Sum() / totalCount;

                var mu0 = (float) class0.Select((count, index) => index * count).Sum() / class0.Sum();
                var mu1 = (float) class1.Select((count, index) => (index + t) * count).Sum() / class1.Sum();

                var interClassVariance = w0 * w1 * Mathf.Pow(mu0 - mu1, 2);
                if (interClassVariance > maxInterClassVariance) {
                    maxInterClassVariance = interClassVariance;
                    threshold = t;
                }
            }

            return threshold;
        }
    }
}