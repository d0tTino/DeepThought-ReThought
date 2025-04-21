using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;

/// <summary>
/// Dispatcher for executing actions on Unity's main thread.
/// Required for safely updating Unity objects from NATS message callbacks which run on different threads.
/// </summary>
public class UnityMainThreadDispatcher : MonoBehaviour
{
    private static UnityMainThreadDispatcher _instance;
    private static readonly object _lock = new object();
    private static Thread _mainThread;
    private readonly Queue<Action> _actionQueue = new Queue<Action>();
    
    /// <summary>
    /// Gets the singleton instance of the dispatcher.
    /// </summary>
    public static UnityMainThreadDispatcher Instance()
    {
        if (_instance == null)
        {
            lock (_lock)
            {
                if (_instance == null)
                {
                    // Find existing instance
                    _instance = FindObjectOfType<UnityMainThreadDispatcher>();
                    
                    // Create new instance if one doesn't already exist
                    if (_instance == null)
                    {
                        // Create gameobject
                        var obj = new GameObject("UnityMainThreadDispatcher");
                        _instance = obj.AddComponent<UnityMainThreadDispatcher>();
                        
                        // Make instance persistent
                        DontDestroyOnLoad(obj);
                    }
                    
                    _mainThread = Thread.CurrentThread;
                }
            }
        }
        
        return _instance;
    }
    
    private void Awake()
    {
        if (_instance == null)
        {
            _instance = this;
            _mainThread = Thread.CurrentThread;
            DontDestroyOnLoad(gameObject);
        }
    }
    
    private void Update()
    {
        // Execute all actions in the queue
        lock (_actionQueue)
        {
            while (_actionQueue.Count > 0)
            {
                Action action = _actionQueue.Dequeue();
                action?.Invoke();
            }
        }
    }
    
    /// <summary>
    /// Determines if the current thread is the main Unity thread.
    /// </summary>
    /// <returns>True if the current thread is the main thread.</returns>
    public bool IsMainThread()
    {
        return Thread.CurrentThread == _mainThread;
    }
    
    /// <summary>
    /// Enqueues an action to be executed on the main thread.
    /// </summary>
    /// <param name="action">Action to be executed on the main thread.</param>
    public void Enqueue(Action action)
    {
        if (action == null)
        {
            Debug.LogWarning("Trying to enqueue a null action");
            return;
        }
        
        // If we're on the main thread, execute immediately
        if (IsMainThread())
        {
            action();
            return;
        }
        
        // Otherwise, add to queue for later execution
        lock (_actionQueue)
        {
            _actionQueue.Enqueue(action);
        }
    }
    
    /// <summary>
    /// Executes an action on the main thread and waits for it to complete.
    /// </summary>
    /// <param name="action">Action to execute on the main thread.</param>
    public void ExecuteSync(Action action)
    {
        if (IsMainThread())
        {
            action();
            return;
        }
        
        // Using a ManualResetEvent to signal when the action has been executed
        using (ManualResetEvent evt = new ManualResetEvent(false))
        {
            Enqueue(() =>
            {
                action();
                evt.Set();
            });
            
            // Wait for the action to be executed
            evt.WaitOne();
        }
    }
    
    /// <summary>
    /// Executes a coroutine on the main thread from a background thread.
    /// </summary>
    /// <param name="routine">The coroutine to start.</param>
    /// <returns>The coroutine.</returns>
    public Coroutine EnqueueCoroutine(IEnumerator routine)
    {
        Coroutine coroutine = null;
        
        ExecuteSync(() =>
        {
            coroutine = StartCoroutine(routine);
        });
        
        return coroutine;
    }
} 