using OpenCvSharp;
using TrtCommon;

namespace yolov10_custom
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
            Yolov10Det yolov8Det = new Yolov10Det("E:\\Model\\yolo\\yolov10s.engine");
            Mat image1 = Cv2.ImRead("E:\\ModelData\\image\\demo_2.jpg");
            List<DetResult> detResults = yolov8Det.Predict(new List<Mat> { image1 });
            Mat re_image1 = Visualize.DrawDetResult(detResults[0], image1);
            Cv2.ImShow("image1", re_image1);
            Cv2.WaitKey(0);
        }
    }
}
