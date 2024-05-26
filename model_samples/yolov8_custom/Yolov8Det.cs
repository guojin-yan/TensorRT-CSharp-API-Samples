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
                Nvinfer.OnnxToEngine(enginePath, 20);
            }
            string path = Path.Combine(Path.GetDirectoryName(enginePath), Path.GetFileNameWithoutExtension(enginePath) + ".engine");
            predictor = new Nvinfer(path);
            InputDims = predictor.GetBindingDimensions("images");

            BatchNum = InputDims.d[0];
        }

        public List<DetResult> Predict(List<Mat> images)
        {
            List<DetResult> returnResults = new List<DetResult>();
            for (int begImgNo = 0; begImgNo < images.Count; begImgNo += BatchNum)
            {
                DateTime start = DateTime.Now;
                int endImgNo = Math.Min(images.Count, begImgNo + BatchNum);
                int batch_num = endImgNo - begImgNo;
                List<Mat> normImgBatch = new List<Mat>();
                Factors = new float[batch_num];
                for (int ino = begImgNo; ino < endImgNo; ino++)
                {
                    Mat mat = new Mat();
                    Cv2.CvtColor(images[ino], mat, ColorConversionCodes.BGR2RGB);
                    mat = Resize.LetterboxImg(mat, InputDims.d[2], out Factors[ino - begImgNo]);
                    mat = Normalize.Run(mat, true);
                    normImgBatch.Add(mat);
                }
                float[] inputData = PermuteBatch.Run(normImgBatch);

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
                List<DetResult> results = ProcessResult(outputData, batch_num);
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

                float[] subArray = new float[705600];

                // 使用Array.Copy方法复制子数组
                Array.Copy(result, 705600 * b, subArray, 0, 705600);

                // Storage results list
                List<Rect> positionBoxes = new List<Rect>();
                List<int> classIds = new List<int>();
                List<float> confidences = new List<float>();
                // Preprocessing output results
                for (int i = 0; i < OutputLength; i++)
                {
                    for (int j = 4; j < 4 + CategNums; j++)
                    {
                        float source = subArray[OutputLength * j + i];
                        int label = j - 4;
                        if (source > DetThresh)
                        {
                            float maxSource = source;
                            float cx = subArray[OutputLength * 0 + i];
                            float cy = subArray[OutputLength * 1 + i];
                            float ow = subArray[OutputLength * 2 + i];
                            float oh = subArray[OutputLength * 3 + i];
                            int x = (int)((cx - 0.5 * ow) * Factors[b]);
                            int y = (int)((cy - 0.5 * oh) * Factors[b]);
                            int width = (int)(ow * Factors[b]);
                            int height = (int)(oh * Factors[b]);
                            Rect box = new Rect(x, y, width, height);
                            positionBoxes.Add(box);
                            classIds.Add(label);
                            confidences.Add(maxSource);
                        }
                    }
                }
                DetResult re = new DetResult();
                int[] indexes = new int[positionBoxes.Count];
                CvDnn.NMSBoxes(positionBoxes, confidences, this.DetThresh, this.DetNmsThresh, out indexes);
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