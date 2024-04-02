using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using TensorRtSharp;
using TensorRtSharp.Custom;
using TrtCommon;

namespace Yolov8
{
    public class Yolov8Seg
    {
        public int CategNums = 80;
        public float DetThresh = 0.5f;
        public float DetNmsThresh = 0.5f;
        public float[] Factors;
        public Dims InputDims;
        public int OutputLength = 8400;
        public int MaskLength = 160;
        public int BatchNum;
        public List<Size> ImageSizes;

        private Nvinfer predictor;
        public Yolov8Seg(string enginePath)
        {
            if (Path.GetExtension(enginePath) == ".onnx") 
            {
                Dims minShapes = new Dims(1, 3, 640, 640);
                Dims optShapes = new Dims(2, 3, 640, 640);
                Dims maxShapes = new Dims(10, 3, 640, 640);
                Nvinfer.OnnxToEngine(enginePath, 20, "images", minShapes, optShapes, maxShapes);
            }
            string path = Path.Combine(Path.GetDirectoryName(enginePath), Path.GetFileNameWithoutExtension(enginePath) + ".engine");
            predictor = new Nvinfer(path,10);
            InputDims = predictor.GetBindingDimensions("images");
        }
        public List<SegResult> Predict(List<Mat> images)
        {
            List<SegResult> returnResults = new List<SegResult>();
            BatchNum = images.Count;
            for (int begImgNo = 0; begImgNo < images.Count; begImgNo += BatchNum)
            {
                DateTime start = DateTime.Now;
                int endImgNo = Math.Min(images.Count, begImgNo + BatchNum);
                int batchNum = endImgNo - begImgNo;
                List<Mat> normImgBatch = new List<Mat>();
                Factors = new float[batchNum];
                ImageSizes = new List<Size>(batchNum);
                for (int ino = begImgNo; ino < endImgNo; ino++)
                {
                    Mat mat = new Mat();
                    ImageSizes.Add(images[ino].Size());
                    Cv2.CvtColor(images[ino], mat, ColorConversionCodes.BGR2RGB);
                    mat = Resize.LetterboxImg(mat, InputDims.d[2], out Factors[ino - begImgNo]);
                    mat = Normalize.Run(mat, true);
                    normImgBatch.Add(mat);
                }
                float[] inputData = PermuteBatch.Run(normImgBatch);
                predictor.SetBindingDimensions("images", new Dims(batchNum, 3, 640, 640));
                predictor.LoadInferenceData("images", inputData);

                DateTime end = DateTime.Now;
                Slog.INFO("Input image data processing time: " + (end - start).TotalMilliseconds + " ms.");
                predictor.infer();
                start = DateTime.Now;
                predictor.infer();
                end = DateTime.Now;
                Slog.INFO("Model inference time: " + (end - start).TotalMilliseconds + " ms.");
                start = DateTime.Now;

                Dims d = predictor.GetBindingDimensions("output0");
                float[] detect = predictor.GetInferenceResult("output0");
                float[] proto = predictor.GetInferenceResult("output1");
                List<SegResult> results = ProcessResult(detect, proto, batchNum);
                end = DateTime.Now;
                Slog.INFO("Inference result processing time: " + (end - start).TotalMilliseconds + " ms.");
                returnResults.AddRange(results);
            }
            return returnResults;

        }

        /// <summary>
        /// Result process
        /// </summary>
        /// <param name="detect">detection output</param>
        /// <param name="proto">segmentation output</param>
        /// <returns></returns>
        public List<SegResult> ProcessResult(float[] detect, float[] proto, int batch)
        {
            List<SegResult> reResult = new List<SegResult>();
            for (int b = 0; b < batch; ++b)
            {
                Mat detectData = new Mat(36 + CategNums, OutputLength, MatType.CV_32FC1,
                     Marshal.UnsafeAddrOfPinnedArrayElement(detect, (4 + CategNums + 32) * OutputLength * b));
                Mat protoData = new Mat(32, MaskLength * MaskLength, MatType.CV_32F,
                    Marshal.UnsafeAddrOfPinnedArrayElement(proto, 32 * MaskLength * MaskLength * b));
                detectData = detectData.T();
                List<Rect> positionBoxes = new List<Rect>();
                List<int> classIds = new List<int>();
                List<float> confidences = new List<float>();
                List<Mat> masks = new List<Mat>();
                for (int i = 0; i < detectData.Rows; i++)
                {

                    Mat classesScores = new Mat(detectData, new Rect(4, i, CategNums, 1));//GetArray(i, 5, classes_scores);
                    Point maxClassIdPoint, min_classId_point;
                    double maxScore, min_score;
                    Cv2.MinMaxLoc(classesScores, out min_score, out maxScore,
                        out min_classId_point, out maxClassIdPoint);

                    if (maxScore > 0.25)
                    {
                        //Console.WriteLine(max_score);

                        Mat mask = new Mat(detectData, new Rect(4 + CategNums, i, 32, 1));//detect_data.Row(i).ColRange(4 + categ_nums, categ_nums + 36);

                        float cx = detectData.At<float>(i, 0);
                        float cy = detectData.At<float>(i, 1);
                        float ow = detectData.At<float>(i, 2);
                        float oh = detectData.At<float>(i, 3);
                        int x = (int)((cx - 0.5 * ow) * this.Factors[b]);
                        int y = (int)((cy - 0.5 * oh) * this.Factors[b]);
                        int width = (int)(ow * this.Factors[b]);
                        int height = (int)(oh * this.Factors[b]);
                        Rect box = new Rect();
                        box.X = x;
                        box.Y = y;
                        box.Width = width;
                        box.Height = height;

                        positionBoxes.Add(box);
                        classIds.Add(maxClassIdPoint.X);
                        confidences.Add((float)maxScore);
                        masks.Add(mask);
                    }
                }


                int[] indexes = new int[positionBoxes.Count];
                CvDnn.NMSBoxes(positionBoxes, confidences, this.DetThresh, this.DetNmsThresh, out indexes);

                SegResult re = new SegResult(); // Output Result Class
                                                // RGB images with colors
                Mat rgbMask = Mat.Zeros(new Size((int)ImageSizes[b].Width, (int)ImageSizes[b].Height), MatType.CV_8UC3);
                Random rd = new Random(); // Generate Random Numbers
                for (int i = 0; i < indexes.Length; i++)
                {
                    int index = indexes[i];
                    // Division scope
                    Rect box = positionBoxes[index];
                    int box_x1 = Math.Max(0, box.X);
                    int box_y1 = Math.Max(0, box.Y);
                    int box_x2 = Math.Max(0, box.BottomRight.X);
                    int box_y2 = Math.Max(0, box.BottomRight.Y);

                    // Segmentation results
                    Mat original_mask = masks[index] * protoData;
                    for (int col = 0; col < original_mask.Cols; col++)
                    {
                        original_mask.Set<float>(0, col, sigmoid(original_mask.At<float>(0, col)));
                    }
                    // 1x25600 -> 160x160 Convert to original size
                    Mat reshapeMask = original_mask.Reshape(1, 160);

                    //Console.WriteLine("m1.size = {0}", m1.Size());

                    // Split size after scaling
                    int mx1 = Math.Max(0, (int)((box_x1 / Factors[b]) * 0.25));
                    int mx2 = Math.Min(160, (int)((box_x2 / Factors[b]) * 0.25));
                    int my1 = Math.Max(0, (int)((box_y1 / Factors[b]) * 0.25));
                    int my2 = Math.Min(160, (int)((box_y2 / Factors[b]) * 0.25));
                    // Crop Split Region
                    Mat maskRoi = new Mat(reshapeMask, new OpenCvSharp.Range(my1, my2), new OpenCvSharp.Range(mx1, mx2));
                    // Convert the segmented area to the actual size of the image
                    Mat actualMaskm = new Mat();
                    Cv2.Resize(maskRoi, actualMaskm, new Size(box_x2 - box_x1, box_y2 - box_y1));
                    // Binary segmentation region
                    for (int r = 0; r < actualMaskm.Rows; r++)
                    {
                        for (int c = 0; c < actualMaskm.Cols; c++)
                        {
                            float pv = actualMaskm.At<float>(r, c);
                            if (pv > 0.5)
                            {
                                actualMaskm.Set<float>(r, c, 1.0f);
                            }
                            else
                            {
                                actualMaskm.Set<float>(r, c, 0.0f);
                            }
                        }
                    }

                    // 预测
                    Mat binMask = new Mat();
                    actualMaskm = actualMaskm * 200;
                    actualMaskm.ConvertTo(binMask, MatType.CV_8UC1);
                    if ((box_y1 + binMask.Rows) >= (int)ImageSizes[b].Height)
                    {
                        box_y2 = (int)ImageSizes[b].Height - 1;
                    }
                    if ((box_x1 + binMask.Cols) >= (int)ImageSizes[b].Width)
                    {
                        box_x2 = (int)ImageSizes[b].Width - 1;
                    }
                    // Obtain segmentation area
                    Mat mask = Mat.Zeros(new Size((int)ImageSizes[b].Width, (int)ImageSizes[b].Height), MatType.CV_8UC1);
                    binMask = new Mat(binMask, new OpenCvSharp.Range(0, box_y2 - box_y1), new OpenCvSharp.Range(0, box_x2 - box_x1));
                    Rect roi = new Rect(box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1);
                    binMask.CopyTo(new Mat(mask, roi));
                    // Color segmentation area
                    Cv2.Add(rgbMask, new Scalar(rd.Next(0, 255), rd.Next(0, 255), rd.Next(0, 255)), rgbMask, mask);

                    re.Add(classIds[index], confidences[index], positionBoxes[index], rgbMask.Clone());

                }
                reResult.Add(re);
            }

            return reResult;
        }
        /// <summary>
        /// sigmoid
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        private float sigmoid(float a)
        {
            float b = 1.0f / (1.0f + (float)Math.Exp(-a));
            return b;
        }

    }
}
