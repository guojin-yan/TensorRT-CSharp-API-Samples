using System;
using System.Collections.Generic;

namespace TrtCommon
{
    public interface IResultData
    {
        string DataToString(string format="0.00");
    }

    /// <summary>
    /// Model inference result class.
    /// </summary>
    /// <typeparam name="T">Model inference result type.</typeparam>
    public class Result<T>
    {
        /// <summary>
        /// Model inference results list.
        /// </summary>
        public List<T> datas;
        /// <summary>
        /// Default Constructor.
        /// </summary>
        public Result()
        {
            datas = new List<T>();
        }
        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="datas">Initialized data.</param>
        public Result(List<T> datas)
        {
            this.datas = datas;
        }
        /// <summary>
        /// Copy Constructor.
        /// </summary>
        /// <param name="values"></param>
        public Result(Result<T> values)
        {
            datas = new List<T>(values.datas);
        }

        /// <summary>
        /// Gets or sets the element at the specified index.
        /// </summary>
        /// <param name="index">The zero-based index of the element to get or set.</param>
        /// <returns>The element at the specified index.</returns>
        public T this[int index] 
        {  
            get { return datas[index]; }
            set { datas[index] = value;}
        }

        /// <summary>
        /// Gets the number of elements contained in the datas.
        /// </summary>
        public int count
        {
            get { return datas.Count; }
        }
        /// <summary>
        /// Gets or sets the total number of elements the internal data structure can hold without resizing.
        /// </summary>
        public int capacity
        {
            get { return datas.Count; }
            set { datas.Capacity = value; }
        }

        /// <summary>
        /// Adds an object to the end of the inference results.
        /// </summary>
        /// <param name="item">The object to be added to the end of the inference results. The
        /// value can be null for reference types.</param>
        public void Add(T item) => datas.Add(item);

        /// <summary>
        ///  Adds the elements of the specified collection to the end of the inference results.
        /// </summary>
        /// <param name="collection">The collection whose elements should be added to the end of the inference results.</param>
        public void AddRange(List<T> collection) => datas.AddRange(collection);
        /// <summary>
        /// Removes all elements from the inference results.
        /// </summary>
        public void Clear() => datas.Clear();
        /// <summary>
        /// Determines whether an element is in the inference results.
        /// </summary>
        /// <param name="item"> The object to locate in the  inference results.</param>
        /// <returns>true if item is found in the inference results; otherwise, false.</returns>
        public bool Contains(T item) => datas.Contains(item);
        /// <summary>
        /// Copies the entire inference results to a compatible one-dimensional array, starting at the specified 
        /// index of the target array.
        /// </summary>
        /// <param name="array">
        /// The one-dimensional System.Array that is the destination of the elements copied
        /// from inference results. The System.Array must have zero-based indexing.
        /// </param>
        /// <param name="arrayIndex">The zero-based index in array at which copying begins.</param>
        /// <exception cref="System.ArgumentNullException">array is null.</exception>
        /// <exception cref="System.ArgumentOutOfRangeException">arrayIndex is less than 0.</exception>
        /// <exception cref="System.ArgumentException">
        /// The number of elements in the source inference results is greater than the available space from arrayIndex 
        /// to the end of the destination array.</exception>
        public void CopyTo(T[] array, int arrayIndex) => datas.CopyTo(array, arrayIndex); 

        /// <summary>
        /// Copies the entire System.Collections.Generic.List`1 to a compatible one-dimensional
        /// array, starting at the beginning of the target array.
        /// </summary>
        /// <param name="array">
        /// The one-dimensional System.Array that is the destination of the elements copied
        /// from System.Collections.Generic.List`1. The System.Array must have zero-based
        /// indexing.
        /// </param>
        public void CopyTo(T[] array) => datas.CopyTo(array);

        /// <summary>
        /// Copies a range of elements from the inference results to a compatible one-dimensional array, starting at 
        /// the specified index of the target array.
        /// </summary>
        /// <param name="index">The zero-based index in the source inference results at which copying begins.</param>
        /// <param name="array">
        /// The one-dimensional System.Array that is the destination of the elements copied
        /// from inference results. The System.Array must have zero-based indexing.
        /// </param>
        /// <param name="arrayIndex">The zero-based index in array at which copying begins.</param>
        /// <param name="count"> The number of elements to copy.</param>
        public void CopyTo(int index, T[] array, int arrayIndex, int count)
            => datas.CopyTo(index, array, arrayIndex, count);


        /// <summary>
        /// Searches for an element that matches the conditions defined by the specified 
        /// predicate, and returns the first occurrence within the entire inference results.
        /// </summary>
        /// <param name="match">The System.Predicate`1 delegate that defines the conditions of the element to search for.</param>
        /// <returns>
        /// The first element that matches the conditions defined by the specified predicate, 
        /// if found; otherwise, the default value for type T.
        /// </returns>
        public T? Find(Predicate<T> match) => datas.Find(match);
        /// <summary>
        /// Retrieves all the elements that match the conditions defined by the specified predicate.
        /// </summary>
        /// <param name="match">The System.Predicate`1 delegate that defines the conditions of the elements to search for.</param>
        /// <returns>
        /// A inference results containing all the elements that match the conditions defined by the specified 
        /// predicate, if found; otherwise, an empty inference results.
        /// </returns>
        public List<T> FindAll(Predicate<T> match) => datas.FindAll(match);

        /// <summary>
        /// Searches for an element that matches the conditions defined by the specified
        /// predicate, and returns the zero-based index of the first occurrence within the
        /// range of elements in the inference results that starts at the
        /// specified index and contains the specified number of elements.
        /// </summary>
        /// <param name="startIndex">The zero-based starting index of the search.</param>
        /// <param name="count">The number of elements in the section to search.</param>
        /// <param name="match">The System.Predicate`1 delegate that defines the conditions of the element to search for.</param>
        /// <returns>
        /// The zero-based index of the first occurrence of an element that matches the conditions
        /// defined by match, if found; otherwise, -1.
        /// </returns>
        public int FindIndex(int startIndex, int count, Predicate<T> match) 
            => datas.FindIndex(startIndex, count, match);
        /// <summary>
        /// Searches for an element that matches the conditions defined by the specified 
        /// predicate, and returns the zero-based index of the first occurrence within the 
        /// range of elements in the inference results that extends from the specified index to the last element.
        /// </summary>
        /// <param name="startIndex">The zero-based starting index of the search.</param>
        /// <param name="match">
        ///  The System.Predicate`1 delegate that defines the conditions of the element to 
        ///  search for.</param>
        /// <returns></returns>
        public int FindIndex(int startIndex, Predicate<T> match)
            => datas.FindIndex(startIndex, match);
        /// <summary>
        /// Searches for an element that matches the conditions defined by the specified
        /// predicate, and returns the zero-based index of the first occurrence within the
        /// entire inference results.
        /// </summary>
        /// <param name="match">The System.Predicate`1 delegate that defines the conditions of the element to
        /// search for.</param>
        /// <returns>
        /// The zero-based index of the first occurrence of an element that matches the conditions
        /// defined by match, if found; otherwise, -1.
        /// </returns>
        public int FindIndex(Predicate<T> match) => datas.FindIndex(match);


        /// <summary>
        /// Searches for an element that matches the conditions defined by the specified
        /// predicate, and returns the last occurrence within the entire inference results.
        /// </summary>
        /// <param name="match">The System.Predicate`1 delegate that defines the conditions of the element to search for.</param>
        /// <returns>
        /// The last element that matches the conditions defined by the specified predicate,
        /// if found; otherwise, the default value for type T.
        /// </returns>
        public T? FindLast(Predicate<T> match) => datas.FindLast(match);

        /// <summary>
        /// Searches for an element that matches the conditions defined by the specified
        /// predicate, and returns the zero-based index of the last occurrence within the
        /// range of elements in the inference results that contains the
        /// specified number of elements and ends at the specified index.
        /// </summary>
        /// <param name="startIndex">The zero-based starting index of the backward search.</param>
        /// <param name="count"> The number of elements in the section to search.</param>
        /// <param name="match">The System.Predicate`1 delegate that defines the conditions of the element to search for.</param>
        /// <returns>
        /// The zero-based index of the last occurrence of an element that matches the conditions
        /// defined by match, if found; otherwise, -1.
        /// </returns>
        public int FindLastIndex(int startIndex, int count, Predicate<T> match) 
            => datas.FindLastIndex(startIndex, count, match);

        /// <summary>
        /// Searches for an element that matches the conditions defined by the specified
        /// predicate, and returns the zero-based index of the last occurrence within the
        /// range of elements in the inference results that extends from
        /// the first element to the specified index.
        /// </summary>
        /// <param name="startIndex">The zero-based starting index of the backward search.</param>
        /// <param name="match">The System.Predicate`1 delegate that defines the conditions of the element to
        /// search for.</param>
        /// <returns>
        /// The zero-based index of the last occurrence of an element that matches the conditions
        /// defined by match, if found; otherwise, -1.
        /// </returns>
        public int FindLastIndex(int startIndex, Predicate<T> match) 
            => datas.FindLastIndex((int)startIndex, match);

        /// <summary>
        /// Searches for an element that matches the conditions defined by the specified
        /// predicate, and returns the zero-based index of the last occurrence within the
        /// entire inference results.
        /// </summary>
        /// <param name="match">The System.Predicate`1 delegate that defines the conditions of the element to
        /// search for.</param>
        /// <returns>
        /// The zero-based index of the last occurrence of an element that matches the conditions
        /// defined by match, if found; otherwise, -1.
        /// </returns>
        public int FindLastIndex(Predicate<T> match)
            => datas.FindLastIndex(match);

        /// <summary>
        /// Performs the specified action on each element of the inference results.
        /// </summary>
        /// <param name="action"> The System.Action`1 delegate to perform on each element of the inference results.</param>
        public void ForEach(Action<T> action)
        {
            datas.ForEach(action);
        }
        /// <summary>
        /// Returns an enumerator that iterates through the inference results.
        /// </summary>
        /// <returns>A System.Collections.Generic.List`1.Enumerator for the  inference results.</returns>
        public List<T>.Enumerator GetEnumerator() 
            => datas.GetEnumerator();
        /// <summary>
        ///  Creates a shallow copy of a range of elements in the source  inference results.
        /// </summary>
        /// <param name="index">The zero-based System.Collections.Generic.List`1 index at which the range starts.</param>
        /// <param name="count">The number of elements in the range.</param>
        /// <returns>A shallow copy of a range of elements in the source inference results.</returns>
        public List<T> GetRange(int index, int count)
            => datas.GetRange(index, count);

        /// <summary>
        /// Searches for the specified object and returns the zero-based index of the first
        /// occurrence within the range of elements in the inference results
        /// that starts at the specified index and contains the specified number of elements.
        /// </summary>
        /// <param name="item">The object to locate in the inference results. The value can be null for reference types.</param>
        /// <param name="index">The zero-based starting index of the search. 0 (zero) is valid in an empty list.</param>
        /// <param name="count">The number of elements in the section to search.</param>
        /// <returns>
        /// The zero-based index of the first occurrence of item within the range of elements
        /// in the inference results that starts at index and contains count
        /// number of elements, if found; otherwise, -1.
        /// </returns>
        public int IndexOf(T item, int index, int count)
            => datas.IndexOf(item, index, count);
        /// <summary>
        /// Searches for the specified object and returns the zero-based index of the first
        /// occurrence within the range of elements in the inference results
        /// that starts at the specified index and contains the specified number of elements.
        /// </summary>
        /// <param name="item">The object to locate in the inference results. The value can be null for reference types.</param>
        /// <param name="index">The zero-based starting index of the search. 0 (zero) is valid in an empty list.</param>
        /// <returns>
        /// The zero-based index of the first occurrence of item within the range of elements
        /// in the inference results that starts at index and contains count
        /// number of elements, if found; otherwise, -1.
        /// </returns>
        public int IndexOf(T item, int index) 
            => datas.IndexOf(item, index);

        /// <summary>
        /// Searches for the specified object and returns the zero-based index of the first
        /// occurrence within the entire inference results.
        /// </summary>
        /// <param name="item">The object to locate in the inference results. The value can be null for reference types.</param>
        /// <returns>
        /// The zero-based index of the first occurrence of item within the entire inference results, 
        /// if found; otherwise, -1.
        /// </returns>
        public int IndexOf(T item) 
            => datas.IndexOf(item);

        /// <summary>
        /// Inserts an element into the System.Collections.Generic.List`1 at the specified index.
        /// </summary>
        /// <param name="index">The zero-based index at which item should be inserted.</param>
        /// <param name="item">The object to insert. The value can be null for reference types.</param>
        public void Insert(int index, T item) => datas.Insert(index, item);

        /// <summary>
        /// Inserts the elements of a collection into the inference results at the specified index.
        /// </summary>
        /// <param name="index">The zero-based index at which the new elements should be inserted.</param>
        /// <param name="collection">
        /// The collection whose elements should be inserted into the inference results.
        /// The collection itself cannot be null, but it can contain elements that are null,
        /// if type T is a reference type.
        /// </param>
        public void InsertRange(int index, IEnumerable<T> collection) 
            => datas.InsertRange(index, collection);

        /// <summary>
        /// Searches for the specified object and returns the zero-based index of the last
        /// occurrence within the entire inference results.
        /// </summary>
        /// <param name="item">The object to locate in the inference results. The value can be null for reference types.</param>
        /// <returns>
        /// The zero-based index of the last occurrence of item within the entire the inference results, if found; otherwise, -1.
        /// </returns>
        public int LastIndexOf(T item)  => datas.LastIndexOf(item);

        /// <summary>
        /// Searches for the specified object and returns the zero-based index of the last
        /// occurrence within the range of elements in the inference results
        /// that contains the specified number of elements and ends at the specified index.
        /// </summary>
        /// <param name="item">The object to locate in the System.Collections.Generic.List`1. The value can
        /// be null for reference types.</param>
        /// <param name="index">The zero-based starting index of the backward search.</param>
        /// <returns>
        /// The zero-based index of the last occurrence of item within the range of elements
        /// in the inference results that contains count number of elements
        /// and ends at index, if found; otherwise, -1.
        /// </returns>
        public int LastIndexOf(T item, int index) 
            => datas.LastIndexOf(item, index);
        /// <summary>
        /// Searches for the specified object and returns the zero-based index of the last
        /// occurrence within the range of elements in the inference results
        /// that contains the specified number of elements and ends at the specified index.
        /// </summary>
        /// <param name="item">The object to locate in the System.Collections.Generic.List`1. The value can
        ///  be null for reference types.</param>
        /// <param name="index">The zero-based starting index of the backward search.</param>
        /// <param name="count">The number of elements in the section to search.</param>
        /// <returns>
        /// The zero-based index of the last occurrence of item within the range of elements
        /// in the inference results that contains count number of elements
        /// and ends at index, if found; otherwise, -1.
        /// </returns>
        public int LastIndexOf(T item, int index, int count) 
            => datas.LastIndexOf((T)item, index, count);

        /// <summary>
        /// Removes the first occurrence of a specific object from the inference results.
        /// </summary>
        /// <param name="item">
        /// The object to remove from the inference results. The value can be null for reference types.
        /// </param>
        /// <returns></returns>
        public bool Remove(T item) => datas.Remove(item);
        /// <summary>
        /// Removes all the elements that match the conditions defined by the specified predicate.
        /// </summary>
        /// <param name="match">The System.Predicate`1 delegate that defines the conditions of the elements to
        /// remove.</param>
        /// <returns>The number of elements removed from the inference results.</returns>
        public int RemoveAll(Predicate<T> match) => datas.RemoveAll(match);
        /// <summary>
        /// Removes the element at the specified index of the inference results.
        /// </summary>
        /// <param name="index"></param>
        public void RemoveAt(int index) => datas.RemoveAt(index);

        /// <summary>
        /// Removes a range of elements from the inference results.
        /// </summary>
        /// <param name="index">The zero-based starting index of the range of elements to remove.</param>
        /// <param name="count">The number of elements to remove.</param>
        public void RemoveRange(int index, int count)  => datas.RemoveRange(index, count);
        /// <summary>
        /// Reverses the order of the elements in the specified range.
        /// </summary>
        /// <param name="index">The zero-based starting index of the range to reverse.</param>
        /// <param name="count">The number of elements in the range to reverse.</param>
        public void Reverse(int index, int count) => datas.Reverse(index, count);
        /// <summary>
        /// Reverses the order of the elements in the entire inference results.
        /// </summary>
        public void Reverse() => datas.Reverse();

        /// <summary>
        /// Sorts the elements in the entire inference resultsw using the specified comparer.
        /// </summary>
        /// <param name="comparer">
        /// The inference results implementation to use when comparing 
        /// elements, or null to use the default comparer inference results.Default.
        /// </param>
        public void Sort(IComparer<T>? comparer) => datas.Sort(comparer);

        /// <summary>
        /// Sorts the elements in the entire inference resultsw using the
        /// specified System.Comparison`1.
        /// </summary>
        /// <param name="comparison">The System.Comparison`1 to use when comparing elements.</param>
        public void Sort(Comparison<T> comparison) => datas.Sort(comparison);
        /// <summary>
        /// Sorts the elements in a range of elements in inference results using the specified comparer.
        /// </summary>
        /// <param name="index">The zero-based starting index of the range to sort.</param>
        /// <param name="count">The length of the range to sort.</param>
        /// <param name="comparer">
        /// The inference results implementation to use when comparing 
        /// elements, or null to use the default comparer inference results.Default.
        /// </param>
        public void Sort(int index, int count, IComparer<T>? comparer) 
            => datas.Sort(index, count, comparer);
        /// <summary>
        ///  Sorts the elements in the entire inference results using the default comparer.
        /// </summary>
        public void Sort() =>datas.Sort();

        /// <summary>
        /// Copies the elements of the inference results to a new array.
        /// </summary>
        /// <returns>An array containing copies of the elements of the inference results.</returns>
        public T[] ToArray() => datas.ToArray();

        /// <summary>
        /// Print  the inference results.
        /// </summary>
        /// <param name="format"></param>
        public virtual void Print(string format = "0.00")  { }
        /// <summary>
        /// Get info message.
        /// </summary>
        /// <param name="msg"></param>
        public static void INFO(string msg) => Console.WriteLine("[INFO]  " + msg);
    }
}
