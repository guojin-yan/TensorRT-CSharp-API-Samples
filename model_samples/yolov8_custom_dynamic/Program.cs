using OpenCvSharp;
using TrtCommon;
using Yolov8;

namespace yolov8_custom_dynamic
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
            //Yolov8Det yolov8Det = new Yolov8Det("E:\\Model\\yolov8\\yolov8s_b.engine");
            //Mat image1 = Cv2.ImRead("E:\\ModelData\\image\\demo_1.jpg");
            //Mat image2 = Cv2.ImRead("E:\\ModelData\\image\\demo_2.jpg");

            //List<DetResult> detResults = yolov8Det.Predict(new List<Mat> { image1, image2, image1, image2, image1, image2, image1, image2 });
            //Mat re_image1 = Visualize.DrawDetResult(detResults[0], image1);
            //Mat re_image2 = Visualize.DrawDetResult(detResults[1], image2);

            //Cv2.ImShow("image1", re_image1);
            //Cv2.ImShow("image2", re_image2);
            //Cv2.WaitKey(0);


            //Yolov8Seg yolov8Seg = new Yolov8Seg("E:\\Model\\yolov8\\yolov8s-seg_b.engine");

            //Mat image1 = Cv2.ImRead("E:\\ModelData\\image\\demo_1.jpg");
            //Mat image2 = Cv2.ImRead("E:\\ModelData\\image\\demo_2.jpg");
            //Mat image3 = Cv2.ImRead("E:\\ModelData\\image\\demo_3.jpg");
            //List<SegResult> segResults = yolov8Seg.Predict(new List<Mat> { image1, image2, image3, image1, image2, image3, image1, image2});
            //Mat re_image1 = Visualize.DrawSegResult(segResults[0], image1);
            //Mat re_image2 = Visualize.DrawSegResult(segResults[1], image2);
            //Mat re_image3 = Visualize.DrawSegResult(segResults[2], image3);
            //Cv2.ImShow("image1", re_image1);
            //Cv2.ImShow("image2", re_image2);
            //Cv2.ImShow("image3", re_image3);
            //Cv2.WaitKey(0);



            //Yolov8Pose yolov8Pose = new Yolov8Pose("E:\\Model\\yolov8\\yolov8s-pose_b.engine");

            //Mat image1 = Cv2.ImRead("E:\\ModelData\\image\\demo_1.jpg");
            //Mat image2 = Cv2.ImRead("E:\\ModelData\\image\\demo_2.jpg");

            //List<PoseResult> poseResults = yolov8Pose.Predict(new List<Mat> { image1, image2, image1, image2, image1, image2, image1, image2 });
            //Mat re_image1 = Visualize.DrawPosesResult(poseResults[0], image1);
            //Mat re_image2 = Visualize.DrawPosesResult(poseResults[1], image2);

            //Cv2.ImShow("image1", re_image1);
            //Cv2.ImShow("image2", re_image2);
            //Cv2.WaitKey(0);



            //Yolov8Cls yolov8Cls = new Yolov8Cls("E:\\Model\\yolov8\\yolov8s-cls_b.engine");

            //Mat image1 = Cv2.ImRead("E:\\ModelData\\image\\demo_4.jpg");
            //Mat image2 = Cv2.ImRead("E:\\ModelData\\image\\demo_5.jpg");

            //List<ClsResult> clsResults = yolov8Cls.Predict(new List<Mat> { image1, image2, image1, image2, image1, image2, image1, image2 });
            ////clsResults[0].Print();
            ////clsResults[1].Print();



            Yolov8Obb yolov8Obb = new Yolov8Obb("E:\\Model\\yolov8\\yolov8s-obb_b.engine");
            Mat image1 = Cv2.ImRead("E:\\ModelData\\image\\plane.png");
            Mat image2 = Cv2.ImRead("E:\\ModelData\\image\\tennis_court.png");

            List<ObbResult> obbResults = yolov8Obb.Predict(new List<Mat> { image1, image2, image1, image2, image1, image2, image1, image2 });
            //Mat re_image1 = Visualize.DrawObbResult(obbResults[0], image1);
            //Mat re_image2 = Visualize.DrawObbResult(obbResults[1], image2);

            //Cv2.ImShow("image1", re_image1);
            //Cv2.ImShow("image2", re_image2);
            //Cv2.WaitKey(0);
        }
    }
}
