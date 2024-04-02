using OpenCvSharp.Dnn;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorRtSharp;
using TrtCommon;
using System.Runtime.InteropServices;
using TensorRtSharp.Custom;

namespace Yolov8
{
    internal class Yolov8Det
    {

        public int CategNums = 80;
        public float DetThresh = 0.5f;
        public float DetNmsThresh = 0.5f;
        public float[] Factors;
        public Dims InputDims;
        public int OutputLength = 8400;
        public int BatchNum;

        private Nvinfer predictor;
        public Yolov8Det(string enginePath) 
        {
            if (Path.GetExtension(enginePath) == ".onnx")
            {
                Dims minShapes = new Dims(1, 3, 640, 640);
                Dims optShapes = new Dims(2, 3, 640, 640);
                Dims maxShapes = new Dims(10, 3, 640, 640);
                Nvinfer.OnnxToEngine(enginePath, 20, "images", minShapes, optShapes, maxShapes);
            }
            string path = Path.Combine(Path.GetDirectoryName(enginePath), Path.GetFileNameWithoutExtension(enginePath) + ".engine");
            predictor = new Nvinfer(path, 10);
            InputDims = predictor.GetBindingDimensions("images");
        }
        public List<DetResult> Predict(List<Mat> images)
        {
            List<DetResult> returnResults = new List<DetResult>();
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

                float[] outputData = predictor.GetInferenceResult("output0");
                List<DetResult> results = ProcessResult(outputData, batchNum);
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
        public List<DetResult> ProcessResult(float[] result, int batch)
        {
            List<DetResult> returnResults = new List<DetResult>();
            for (int b = 0; b < batch; ++b)
            {
                Mat resultData = new Mat(4 + CategNums, 8400, MatType.CV_32F,
                    Marshal.UnsafeAddrOfPinnedArrayElement(result, (4 + CategNums) * OutputLength * b), 4 * OutputLength);
                resultData = resultData.T();

                // Storage results list
                List<Rect> positionBoxes = new List<Rect>();
                List<int> classIds = new List<int>();
                List<float> confidences = new List<float>();
                // Preprocessing output results
                for (int i = 0; i < resultData.Rows; i++)
                {
                    Mat classesScores = new Mat(resultData, new Rect(4, i, CategNums, 1));
                    Point maxClassIdPoint, minClassIdPoint;
                    double maxScore, minScore;
                    // Obtain the maximum value and its position in a set of data
                    Cv2.MinMaxLoc(classesScores, out minScore, out maxScore,
                        out minClassIdPoint, out maxClassIdPoint);
                    // Confidence level between 0 ~ 1
                    // Obtain identification box information
                    if (maxScore > 0.25)
                    {
                        float cx = resultData.At<float>(i, 0);
                        float cy = resultData.At<float>(i, 1);
                        float ow = resultData.At<float>(i, 2);
                        float oh = resultData.At<float>(i, 3);
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
                    }
                }
                // NMS non maximum suppression
                int[] indexes = new int[positionBoxes.Count];
                CvDnn.NMSBoxes(positionBoxes, confidences, this.DetThresh, this.DetNmsThresh, out indexes);
                DetResult re = new DetResult();
                // 
                for (int i = 0; i < indexes.Length; i++)
                {
                    int index = indexes[i];
                    re.Add(classIds[index], confidences[index], positionBoxes[index]);
                }
                returnResults.Add(re);
            }
            return returnResults;
        }
    }
}
