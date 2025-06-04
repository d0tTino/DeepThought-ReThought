using System;
using System.Collections;
using System.Threading.Tasks;
using NATS.Client;
using NATS.Client.JetStream;
using UnityEngine;

/// <summary>
/// Manages JetStream configuration for Unity applications.
/// Similar to the Python setup_jetstream.py script but implemented for Unity/C#.
/// </summary>
public class NatsJetStreamManager : MonoBehaviour
{
    [SerializeField] private NatsService _natsService;
    [SerializeField] private string _streamName = "deepthought_events";
    [SerializeField] private string _subjectFilter = "dtr.>";
    [SerializeField] private bool _setupOnStart = true;
    [SerializeField] private int _maxMessagesPerSubject = 10000;
    
    private IJetStreamManagement _jsm;
    
    /// <summary>
    /// Gets a value indicating whether JetStream has been set up.
    /// </summary>
    public bool IsSetup { get; private set; }
    
    /// <summary>
    /// Event triggered when JetStream setup completes.
    /// </summary>
    public event Action<bool> OnSetupComplete;

    private void Start()
    {
        // If NatsService isn't assigned in the inspector, try to find it
        if (_natsService == null)
        {
            _natsService = FindObjectOfType<NatsService>();
            if (_natsService == null)
            {
                Debug.LogError("NatsService not found in the scene!");
                return;
            }
        }
        
        // Subscribe to connection status changes
        _natsService.OnConnectionStatusChanged += HandleConnectionStatusChanged;
        
        if (_setupOnStart && _natsService.IsConnected)
        {
            SetupJetStream();
        }
    }
    
    private void OnDestroy()
    {
        if (_natsService != null)
        {
            _natsService.OnConnectionStatusChanged -= HandleConnectionStatusChanged;
        }
    }
    
    /// <summary>
    /// Handles connection status changes from the NATS service.
    /// </summary>
    /// <param name="isConnected">Whether the client is connected.</param>
    private void HandleConnectionStatusChanged(bool isConnected)
    {
        if (isConnected && _setupOnStart)
        {
            SetupJetStream();
        }
        else if (!isConnected)
        {
            IsSetup = false;
            _jsm = null;
        }
    }
    
    /// <summary>
    /// Sets up JetStream streams for the application.
    /// </summary>
    public void SetupJetStream()
    {
        StartCoroutine(SetupJetStreamAsync());
    }
    
    /// <summary>
    /// Asynchronous coroutine to set up JetStream streams.
    /// </summary>
    private IEnumerator SetupJetStreamAsync()
    {
        Debug.Log("Setting up JetStream streams...");
        
        if (!_natsService.IsConnected)
        {
            Debug.LogError("Cannot set up JetStream: NATS service not connected");
            OnSetupComplete?.Invoke(false);
            yield break;
        }
        
        var task = Task.Run(() => 
        {
            try
            {
                // Get the JetStream management context from the connection
                var connection = (Connection)typeof(NatsService)
                    .GetField("_connection", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                    .GetValue(_natsService);
                
                _jsm = connection.CreateJetStreamManagementContext();
                
                // Configure the stream
                var config = StreamConfiguration.Builder()
                    .WithName(_streamName)
                    .WithSubjects(_subjectFilter)
                    .WithStorageType(StorageType.Memory)
                    .WithRetentionPolicy(RetentionPolicy.Limits)
                    .WithMaxMessagesPerSubject(_maxMessagesPerSubject)
                    .WithDiscardPolicy(DiscardPolicy.Old)
                    .Build();
                
                try
                {
                    // Try to create the stream
                    _jsm.AddStream(config);
                    Debug.Log($"Created JetStream stream: {_streamName}");
                }
                catch (NATSJetStreamException)
                {
                    // If the stream already exists, update it
                    _jsm.UpdateStream(config);
                    Debug.Log($"Updated JetStream stream: {_streamName}");
                }
                
                return true;
            }
            catch (Exception ex)
            {
                UnityMainThreadDispatcher.Instance().Enqueue(() => 
                {
                    Debug.LogError($"Failed to set up JetStream: {ex.Message}");
                });
                return false;
            }
        });
        
        // Wait for the task to complete
        while (!task.IsCompleted)
        {
            yield return null;
        }
        
        bool success = task.Result;
        IsSetup = success;
        OnSetupComplete?.Invoke(success);
        
        if (success)
        {
            Debug.Log("JetStream setup completed successfully");
        }
    }
    
    /// <summary>
    /// Deletes a JetStream stream.
    /// </summary>
    /// <param name="streamName">Name of the stream to delete. If null, uses the default stream name.</param>
    public void DeleteStream(string streamName = null)
    {
        if (_jsm == null)
        {
            Debug.LogError("Cannot delete stream: JetStream management context not available");
            return;
        }
        
        string stream = streamName ?? _streamName;
        
        try
        {
            _jsm.DeleteStream(stream);
            Debug.Log($"Deleted JetStream stream: {stream}");
            
            if (stream == _streamName)
            {
                IsSetup = false;
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to delete stream '{stream}': {ex.Message}");
        }
    }
    
    /// <summary>
    /// Gets information about a JetStream stream.
    /// </summary>
    /// <param name="streamName">Name of the stream to get info for. If null, uses the default stream name.</param>
    /// <returns>A description of the stream or null if not found.</returns>
    public string GetStreamInfo(string streamName = null)
    {
        if (_jsm == null)
        {
            Debug.LogError("Cannot get stream info: JetStream management context not available");
            return null;
        }
        
        string stream = streamName ?? _streamName;
        
        try
        {
            var info = _jsm.GetStreamInfo(stream);
            string infoText = $"Stream: {info.Config.Name}\n" +
                             $"Subjects: {string.Join(", ", info.Config.Subjects)}\n" +
                             $"Messages: {info.State.Messages}\n" +
                             $"Bytes: {info.State.Bytes}\n" +
                             $"Consumer Count: {info.State.ConsumerCount}";
            
            Debug.Log(infoText);
            return infoText;
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to get stream info for '{stream}': {ex.Message}");
            return null;
        }
    }
} 
