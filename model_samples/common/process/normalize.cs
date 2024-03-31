using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrtCommon
{
    /// <summary>
    /// Normalize data classes using OpenCvSharp.
    /// </summary>
    public static class Normalize
    {
        /// <summary>
        /// Run normalize data classes.
        /// </summary>
        /// <param name="im">The image mat.</param>
        /// <param name="mean">Channel mean.</param>
        /// <param name="scale">Channel variance.</param>
        /// <param name="isScale">Whether to divide by 255.</param>
        /// <returns>The normalize data.</returns>
        public static Mat Run(Mat im, float[] mean, float[] scale, bool isScale)
        {
            double e = 1.0;
            if (isScale)
            {
                e /= 255.0;
            }
            im.ConvertTo(im, MatType.CV_32FC3, e);
            Mat[] bgrChannels = new Mat[3];
            Cv2.Split(im, out bgrChannels);
            for (var i = 0; i < bgrChannels.Length; i++)
            {
                bgrChannels[i].ConvertTo(bgrChannels[i], MatType.CV_32FC1, 1.0 * scale[i],
                    (0.0 - mean[i]) * scale[i]);
            }
            Mat re = new Mat();
            Cv2.Merge(bgrChannels, re);
            return re;
        }
        /// <summary>
        /// Run normalize data classes.
        /// </summary>
        /// <param name="im">The image mat.</param>
        /// <param name="isScale">Whether to divide by 255.</param>
        /// <returns>The normalize data.</returns>
        public static Mat Run(Mat im, bool isScale)
        {
            double e = 1.0;
            if (isScale)
            {
                e /= 255.0;
            }
            im.ConvertTo(im, MatType.CV_32FC3, e);
            return im;
        }

    }
}
