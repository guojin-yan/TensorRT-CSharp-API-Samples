using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorRtSharp.Custom;
using TensorRtSharp;
using TrtCommon;
using OpenCvSharp.Dnn;
using System.Runtime.InteropServices;
using System.Numerics;

namespace Yolov8
{
    internal class Yolov8Obb
    {
        public int CategNums = 15;
        public float DetThresh = 0.5f;
        public float DetNmsThresh = 0.5f;
        public float[] Factors;
        public Dims InputDims;
        public int OutputLength = 21504;
        public int BatchNum;

        private Nvinfer predictor;
        public Yolov8Obb(string enginePath)
        {
            if (Path.GetExtension(enginePath) == ".onnx")
            {
                Dims minShapes = new Dims(1, 3, 1024, 1024);
                Dims optShapes = new Dims(2, 3, 1024, 1024);
                Dims maxShapes = new Dims(10, 3, 1024, 1024);
                Nvinfer.OnnxToEngine(enginePath, 20, "images", minShapes, optShapes, maxShapes);
            }
            string path = Path.Combine(Path.GetDirectoryName(enginePath), Path.GetFileNameWithoutExtension(enginePath) + ".engine");
            predictor = new Nvinfer(path,10);
            InputDims = predictor.GetBindingDimensions("images");
        }
        public List<ObbResult> Predict(List<Mat> images)
        {
            List<ObbResult> returnResults = new List<ObbResult>();
            BatchNum = images.Count;
            for (int begImgNo = 0; begImgNo < images.Count; begImgNo += BatchNum)
            {
                DateTime start = DateTime.Now;
                int endImgNo = Math.Min(images.Count, begImgNo + BatchNum);
                int batchNum = endImgNo - begImgNo;
                List<Mat> normImgBatch = new List<Mat>();
                Factors = new float[batchNum];
                for (int ino = begImgNo; ino < endImgNo; ino++)
                {
                    Mat mat = new Mat();
                    Cv2.CvtColor(images[ino], mat, ColorConversionCodes.BGR2RGB);
                    mat = Resize.LetterboxImg(mat, InputDims.d[2], out Factors[ino - begImgNo]);
                    mat = Normalize.Run(mat, true);
                    normImgBatch.Add(mat);
                }
                float[] inputData = PermuteBatch.Run(normImgBatch);
                predictor.SetBindingDimensions("images", new Dims(batchNum, 3, 1024, 1024));
                predictor.LoadInferenceData("images", inputData);
                DateTime end = DateTime.Now;
                Slog.INFO("Input image data processing time: " + (end - start).TotalMilliseconds + " ms.");
                predictor.infer();
                start = DateTime.Now;
                predictor.infer();
                end = DateTime.Now;
                Slog.INFO("Model inference time: " + (end - start).TotalMilliseconds + " ms.");
                start = DateTime.Now;
                Dims dims = predictor.GetBindingDimensions("output0");
                float[] outputData = predictor.GetInferenceResult("output0");
                List<ObbResult> results = ProcessResult(outputData, batchNum);
                end = DateTime.Now;
                Slog.INFO("Inference result processing time: " + (end - start).TotalMilliseconds + " ms.");
                returnResults.AddRange(results);
            }
            return returnResults;

        }

        /// <summary>
        /// Result process
        /// </summary>
        /// <param name="result">Model prediction output</param>
        /// <returns>Model recognition results</returns>
        public List<ObbResult> ProcessResult(float[] result, int batch)
        {
            List<ObbResult> returnResults = new List<ObbResult>();
            for (int b = 0; b < batch; ++b)
            {
                Mat resultData = new Mat(5 + CategNums, OutputLength, MatType.CV_32F,
                    Marshal.UnsafeAddrOfPinnedArrayElement(result, (5 + CategNums) * OutputLength * b), 4 * OutputLength);
                resultData = resultData.T();

                // Storage results list
                List<Rect2d> positionBoxes = new List<Rect2d>();
                List<int> classIds = new List<int>();
                List<float> confidences = new List<float>();
                List<float> rotations = new List<float>();
                // Preprocessing output results
                for (int i = 0; i < resultData.Rows; i++)
                {
                    Mat classesScores = new Mat(resultData, new Rect(4, i, 15, 1));
                    OpenCvSharp.Point max_classId_point, min_classId_point;
                    double maxScore, minScore;
                    // Obtain the maximum value and its position in a set of data
                    Cv2.MinMaxLoc(classesScores, out minScore, out maxScore,
                        out min_classId_point, out max_classId_point);
                    // Confidence level between 0 ~ 1
                    // Obtain identification box information
                    if (maxScore > 0.25)
                    {
                        float cx = resultData.At<float>(i, 0);
                        float cy = resultData.At<float>(i, 1);
                        float ow = resultData.At<float>(i, 2);
                        float oh = resultData.At<float>(i, 3);
                        double x = (cx - 0.5 * ow) * Factors[b];
                        double y = (cy - 0.5 * oh) * Factors[b];
                        double width = ow * Factors[b];
                        double height = oh * Factors[b];
                        Rect2d box = new Rect2d();
                        box.X = x;
                        box.Y = y;
                        box.Width = width;
                        box.Height = height;

                        positionBoxes.Add(box);
                        classIds.Add(max_classId_point.X);
                        confidences.Add((float)maxScore);
                        rotations.Add(resultData.At<float>(i, 19));
                    }
                }
                // NMS non maximum suppression
                int[] indexes = new int[positionBoxes.Count];
                CvDnn.NMSBoxes(positionBoxes, confidences, DetThresh, DetNmsThresh, out indexes);

                ObbResult obbResult = new ObbResult();
                for (int i = 0; i < indexes.Length; i++)
                {
                    int index = indexes[i];
                    float w = (float)positionBoxes[index].Width;
                    float h = (float)positionBoxes[index].Height;
                    float x = (float)positionBoxes[index].X + w / 2;
                    float y = (float)positionBoxes[index].Y + h / 2;
                    float r = rotations[index];
                    float w_ = w > h ? w : h;
                    float h_ = w > h ? h : w;
                    r = (float)((w > h ? r : (float)(r + Math.PI / 2)) % Math.PI);
                    RotatedRect rotate = new RotatedRect(new Point2f(x, y), new Size2f(w_, h_), (float)(r * 180.0 / Math.PI));
                    obbResult.Add(classIds[index], confidences[index], rotate);
                }

                returnResults.Add(obbResult);
            }
            return returnResults;
        }
    }
}
