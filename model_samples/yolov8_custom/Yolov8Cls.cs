using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorRtSharp;
using TensorRtSharp.Custom;
using TrtCommon;

namespace Yolov8
{
    public class Yolov8Cls
    {
        public int CategNums = 1000;
        public Dims InputDims;
        public int BatchNum;
        public int ResultNum = 10;
        private Nvinfer predictor;

        public Yolov8Cls(string enginePath)
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


        public List<ClsResult> Predict(List<Mat> images)
        {
            List<ClsResult> results = new List<ClsResult>();
            for (int begImgNo = 0; begImgNo < images.Count; begImgNo += BatchNum)
            {
                DateTime start = DateTime.Now;
                int endImgNo = Math.Min(images.Count, begImgNo + BatchNum);
                int batchNum = endImgNo - begImgNo;
                List<Mat> normImgBatch = new List<Mat>();
                float factors = 0f;
                for (int ino = begImgNo; ino < endImgNo; ino++)
                {
                    Mat mat = new Mat();
                    Cv2.CvtColor(images[ino], mat, ColorConversionCodes.BGR2RGB);
                    mat = Resize.LetterboxImg(mat, InputDims.d[2], out factors);
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
                for (int i = 0; i < batchNum; ++i)
                {
                    float[] data = new float[CategNums];
                    Array.Copy(outputData, i * CategNums, data, 0, CategNums);
                    List<int> sortResult = Utility.Argsort(data);
                    ClsResult result = new ClsResult();
                    for (int j = 0; j < ResultNum; ++j)
                    {
                        result.Add(sortResult[j], data[sortResult[j]]);
                    }
                    results.Add(result);
                }
                end = DateTime.Now;
                Slog.INFO("Inference result processing time: " + (end - start).TotalMilliseconds + " ms.");
            }
            return results;
        }
    }
}
