using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrtCommon
{
    public static partial class Utility
    {
        /// <summary>
        /// Obtain the original position of the arranged array.
        /// </summary>
        /// <param name="array">The original array.</param>
        /// <returns>The position after arrangement.</returns>
        public static List<int> Argsort(List<float> array)
        {
            int arrayLen = array.Count;

            //生成值和索引的列表
            List<float[]> newArray = new List<float[]> { };
            for (int i = 0; i < arrayLen; i++)
            {
                newArray.Add(new float[] { array[i], i });
            }
            //对列表按照值大到小进行排序
            newArray.Sort((a, b) => b[0].CompareTo(a[0]));
            //获取排序后的原索引
            List<int> arrayIndex = new List<int>();
            foreach (float[] item in newArray)
            {
                arrayIndex.Add((int)item[1]);
            }
            return arrayIndex;
        }
        /// <summary>
        /// Obtain the original position of the arranged array.
        /// </summary>
        /// <param name="array">The original array.</param>
        /// <returns>The position after arrangement.</returns>
        public static List<int> Argsort(float[] array) 
        {
            return Argsort(new List<float>(array));
        }
    }
}
