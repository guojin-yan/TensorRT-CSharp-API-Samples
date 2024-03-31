using OpenCvSharp.Dnn;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using TrtCommon;
using TensorRtSharp.Custom;
using TensorRtSharp;

namespace Yolov8
{
    public class Yolov8Pose
    {
        public float DetThresh = 0.5f;
        public float DetNmsThresh = 0.5f;
        public float[] Factors;
        public Dims InputDims;
        public int BatchNum;
        public int OutputLength = 8400;

        private Nvinfer predictor;

        public Yolov8Pose(string enginePath)
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
        public List<PoseResult> Predict(List<Mat> images)
        {
            List<PoseResult> reResults = new List<PoseResult>();
            for (int begImgNo = 0; begImgNo < images.Count; begImgNo += BatchNum)
            {
                DateTime start = DateTime.Now;
                int endImgNo = Math.Min(images.Count, begImgNo + this.BatchNum);
                int batchNum = endImgNo - begImgNo;
                List<Mat> normImgBatch = new List<Mat>();
                Factors = new float[batchNum];
                for (int ino = begImgNo; ino < endImgNo; ino++)
                {
                    Mat mat = new Mat();
                    Cv2.CvtColor(images[ino], mat, ColorConversionCodes.BGR2RGB);
                    mat = Resize.LetterboxImg(mat, (int)InputDims.d[2], out Factors[ino - begImgNo]);
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
                List<PoseResult> results = ProcessResult(outputData, batchNum);
                end = DateTime.Now;
                Slog.INFO("Inference result processing time: " + (end - start).TotalMilliseconds + " ms.");
                reResults.AddRange(results);
            }
            return reResults;

        }

        /// <summary>
        /// Result process
        /// </summary>
        /// <param name="result">Model prediction output</param>
        /// <returns>Model recognition results</returns>
        public List<PoseResult> ProcessResult(float[] result, int batch)
        {
            List<PoseResult> reResult = new List<PoseResult>();
            for (int b = 0; b < batch; ++b)
            {
                Mat resultData = new Mat(56, OutputLength, MatType.CV_32FC1,
                    Marshal.UnsafeAddrOfPinnedArrayElement(result, 56 * OutputLength * b));
                resultData = resultData.T();
                List<Rect> positionBoxes = new List<Rect>();
                List<float> confidences = new List<float>();
                List<PosePoint> poseDatas = new List<PosePoint>();
                for (int i = 0; i < resultData.Rows; i++)
                {
                    if (resultData.At<float>(i, 4) > 0.25)
                    {
                        //Console.WriteLine(max_score);
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
                        Mat poseMat = new Mat(resultData, new Rect(5, i, 51, 1));//result_data.Row(i).ColRange(5, 56);
                        IntPtr pt = poseMat.Data;
                        float[] poseData = new float[51];
                        Marshal.Copy(pt, poseData, 0, poseData.Length);
                        PosePoint pose = new PosePoint(poseData, this.Factors[b]);

                        positionBoxes.Add(box);

                        confidences.Add((float)resultData.At<float>(i, 4));
                        poseDatas.Add(pose);
                    }
                }

                int[] indexes = new int[positionBoxes.Count];
                CvDnn.NMSBoxes(positionBoxes, confidences, this.DetThresh, this.DetNmsThresh, out indexes);

                PoseResult re = new PoseResult();
                for (int i = 0; i < indexes.Length; i++)
                {
                    int index = indexes[i];
                    re.Add(confidences[index], positionBoxes[index], poseDatas[index]);
                    //Console.WriteLine("rect: {0}, score: {1}", position_boxes[index], confidences[index]);
                }
                reResult.Add(re);
            }

            return reResult;

        }
    }
}
