using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Size = OpenCvSharp.Size;

namespace TrtCommon
{
    public static class Resize
    {
        public static Mat ImgType0(Mat img, string limitType, int limitSideLen,
            out float ratioH, out float ratioW)
        {
            int w = img.Cols;
            int h = img.Rows;
            float ratio = 1.0f;
            if (limitType == "min")
            {
                int minWH = Math.Min(h, w);
                if (minWH < limitSideLen)
                {
                    if (h < w)
                    {
                        ratio = (float)limitSideLen / (float)h;
                    }
                    else
                    {
                        ratio = (float)limitSideLen / (float)w;
                    }
                }
            }
            else
            {
                int maxWH = Math.Max(h, w);
                if (maxWH > limitSideLen)
                {
                    if (h > w)
                    {
                        ratio = (float)(limitSideLen) / (float)(h);
                    }
                    else
                    {
                        ratio = (float)(limitSideLen) / (float)(w);
                    }
                }
            }

            int resizeH = (int)((float)(h) * ratio);
            int resizeW = (int)((float)(w) * ratio);

            //int resize_h = 960;
            //int resize_w = 960;

            resizeH = Math.Max((int)(Math.Round((float)(resizeH) / 32.0f) * 32), 32);
            resizeW = Math.Max((int)(Math.Round((float)(resizeW) / 32.0f) * 32), 32);

            Mat resizeImg = new Mat();
            Cv2.Resize(img, resizeImg, new Size(resizeW, resizeH));
            ratioH = (float)(resizeH) / (float)(h);
            ratioW = (float)(resizeW) / (float)(w);
            return resizeImg;
        }

        public static Mat ClsImg(Mat img, List<int> clsImgShape)
        {
            int imgC, imgH, imgW;
            imgC = clsImgShape[0];
            imgH = clsImgShape[1];
            imgW = clsImgShape[2];

            float ratio = (float)img.Cols / (float)img.Rows;
            int resizeW, resizeH;
            if (Math.Ceiling(imgH * ratio) > imgW)
                resizeW = imgW;
            else
                resizeW = (int)(Math.Ceiling(imgH * ratio));
            Mat resizeImg = new Mat();
            Cv2.Resize(img, resizeImg, new Size(resizeW, imgH), 0.0f, 0.0f, InterpolationFlags.Linear);
            return resizeImg;
        }


        public static Mat CrnnImg(Mat img, float whRatio, int[] recImgShape)
        {
            int imgC, imgH, imgW;
            imgC = recImgShape[0];
            imgH = recImgShape[1];
            imgW = recImgShape[2];

            imgW = (int)(imgH * whRatio);

            float ratio = (float)(img.Cols) / (float)(img.Rows);
            int resizeW, resizeH;

            if (Math.Ceiling(imgH * ratio) > imgW)
                resizeW = imgW;
            else
                resizeW = (int)(Math.Ceiling(imgH * ratio));
            Mat resizeImg = new Mat();
            Cv2.Resize(img, resizeImg, new Size(resizeW, imgH), 0.0f, 0.0f, InterpolationFlags.Linear);
            Cv2.CopyMakeBorder(resizeImg, resizeImg, 0, 0, 0, (int)(imgW - resizeImg.Cols), BorderTypes.Constant, new Scalar(127, 127, 127));
            return resizeImg;
        }

        public static Mat LetterboxImg(Mat image, int length, out float scales) 
        {
            int maxImageLength = image.Cols > image.Rows ? image.Cols : image.Rows;
            Mat maxImage = Mat.Zeros(maxImageLength, maxImageLength, MatType.CV_8UC3);
            maxImage = maxImage * 255;
            Rect roi = new Rect(0, 0, image.Cols, image.Rows);
            image.CopyTo(new Mat(maxImage, roi));
            Mat resizeImg = new Mat();
            Cv2.Resize(maxImage, resizeImg, new Size(length, length), 0.0f, 0.0f, InterpolationFlags.Linear);
            scales = (float)((float)maxImageLength / (float)length);
            return resizeImg;
        }

    }
}
