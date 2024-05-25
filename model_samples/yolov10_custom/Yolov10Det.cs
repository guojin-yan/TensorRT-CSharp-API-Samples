using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using TensorRtSharp.Custom;
using TensorRtSharp;
using TrtCommon;
using OpenCvSharp;
using System.Numerics;
using OpenCvSharp.Flann;

namespace yolov10_custom
{
    internal class Yolov10Det
    {
        public int CategNums = 80;
        public float DetThresh = 0.5f;
        public float DetNmsThresh = 0.5f;
        public float[] Factors;
        public Dims InputDims;
        public int OutputLength = 8400;
        public int BatchNum;

        private Nvinfer predictor;
        public Yolov10Det(string enginePath)
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
        public List<DetResult> ProcessResult(float[] results, int batch)
        {
            List<DetResult> returnResults = new List<DetResult>();
            for (int b = 0; b < batch; ++b)
            {
                float[] output_data = new float[results.Length/batch];
                Array.Copy(results, results.Length / batch * b, output_data, 0, results.Length / batch);
                List<Rect> positionBoxes = new List<Rect>();
                List<int> classIds = new List<int>();
                List<float> confidences = new List<float>();

                // Preprocessing output results
                for (int i = 0; i < output_data.Length / 6; i++)
                {
                    int s = 6 * i;
                    if ((float)output_data[s + 4] > 0.5)
                    {
                        float cx = output_data[s + 0];
                        float cy = output_data[s + 1];
                        float dx = output_data[s + 2];
                        float dy = output_data[s + 3];
                        int x = (int)((cx) * Factors[b]);
                        int y = (int)((cy) * Factors[b]);
                        int width = (int)((dx - cx) * Factors[b]);
                        int height = (int)((dy - cy) * Factors[b]);
                        Rect box = new Rect();
                        box.X = x;
                        box.Y = y;
                        box.Width = width;
                        box.Height = height;

                        positionBoxes.Add(box);
                        classIds.Add((int)output_data[s + 5]);
                        confidences.Add((float)output_data[s + 4]);
                    }
                }
                DetResult re = new DetResult();
                // 
                for (int i = 0; i < positionBoxes.Count; i++)
                {
                    re.Add(classIds[i], confidences[i], positionBoxes[i]);
                }
                returnResults.Add(re);
            }
            return returnResults;
        }
    }
}
