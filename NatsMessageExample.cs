using System.Collections;
using UnityEngine;
using NATS.Client;

/// <summary>
/// Example component demonstrating how to use the NatsService for messaging.
/// </summary>
public class NatsMessageExample : MonoBehaviour
{
    [SerializeField] private NatsService _natsService;
    [SerializeField] private string _publishSubject = "dtr.example";
    [SerializeField] private string _subscribeSubject = "dtr.example";
    [SerializeField] private float _publishInterval = 5f;
    
    private IAsyncSubscription _subscription;
    private bool _isPublishing = false;
    private Coroutine _publishCoroutine;

    /// <summary>
    /// Message structure for example messages.
    /// </summary>
    public class ExampleMessage
    {
        public string Id { get; set; }
        public string Content { get; set; }
        public long Timestamp { get; set; }
    }

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
        
        // If already connected, subscribe immediately
        if (_natsService.IsConnected)
        {
            SubscribeToMessages();
        }
    }
    
    private void OnDestroy()
    {
        if (_natsService != null)
        {
            _natsService.OnConnectionStatusChanged -= HandleConnectionStatusChanged;
        }
        
        StopPublishing();
        UnsubscribeFromMessages();
    }
    
    /// <summary>
    /// Handles connection status changes from the NATS service.
    /// </summary>
    /// <param name="isConnected">Whether the client is connected.</param>
    private void HandleConnectionStatusChanged(bool isConnected)
    {
        if (isConnected)
        {
            Debug.Log("Connected to NATS server, setting up subscription");
            SubscribeToMessages();
        }
        else
        {
            Debug.Log("Disconnected from NATS server");
            _subscription = null;
        }
    }
    
    /// <summary>
    /// Subscribes to the configured subject.
    /// </summary>
    public void SubscribeToMessages()
    {
        if (_natsService == null || !_natsService.IsConnected)
        {
            Debug.LogWarning("Cannot subscribe: NATS service not available or not connected");
            return;
        }
        
        // Unsubscribe if already subscribed
        UnsubscribeFromMessages();
        
        // Create new subscription
        _subscription = _natsService.Subscribe(_subscribeSubject, HandleMessageReceived);
        Debug.Log($"Subscribed to {_subscribeSubject}");
    }
    
    /// <summary>
    /// Unsubscribes from the current subscription.
    /// </summary>
    public void UnsubscribeFromMessages()
    {
        if (_subscription != null)
        {
            try
            {
                _subscription.Unsubscribe();
                Debug.Log($"Unsubscribed from {_subscribeSubject}");
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"Error unsubscribing: {ex.Message}");
            }
            _subscription = null;
        }
    }
    
    /// <summary>
    /// Starts publishing messages at regular intervals.
    /// </summary>
    public void StartPublishing()
    {
        if (_isPublishing)
            return;
            
        if (_natsService == null || !_natsService.IsConnected)
        {
            Debug.LogWarning("Cannot start publishing: NATS service not available or not connected");
            return;
        }
        
        _isPublishing = true;
        _publishCoroutine = StartCoroutine(PublishRoutine());
        Debug.Log("Started publishing messages");
    }
    
    /// <summary>
    /// Stops publishing messages.
    /// </summary>
    public void StopPublishing()
    {
        if (!_isPublishing)
            return;
            
        if (_publishCoroutine != null)
        {
            StopCoroutine(_publishCoroutine);
            _publishCoroutine = null;
        }
        
        _isPublishing = false;
        Debug.Log("Stopped publishing messages");
    }
    
    /// <summary>
    /// Coroutine to publish messages at regular intervals.
    /// </summary>
    private IEnumerator PublishRoutine()
    {
        while (_isPublishing && _natsService != null && _natsService.IsConnected)
        {
            PublishExampleMessage();
            yield return new WaitForSeconds(_publishInterval);
        }
        
        _isPublishing = false;
    }
    
    /// <summary>
    /// Publishes a single example message.
    /// </summary>
    public void PublishExampleMessage()
    {
        if (_natsService == null || !_natsService.IsConnected)
        {
            Debug.LogWarning("Cannot publish: NATS service not available or not connected");
            return;
        }
        
        var message = new ExampleMessage
        {
            Id = System.Guid.NewGuid().ToString(),
            Content = $"Test message from Unity at {System.DateTime.Now}",
            Timestamp = System.DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
        };
        
        bool success = _natsService.Publish(_publishSubject, message);
        if (success)
        {
            Debug.Log($"Published message to {_publishSubject}: {message.Content}");
        }
    }
    
    /// <summary>
    /// Handles received NATS messages.
    /// </summary>
    private void HandleMessageReceived(object sender, MsgHandlerEventArgs e)
    {
        try
        {
            // Process on the main thread
            UnityMainThreadDispatcher.Instance().Enqueue(() =>
            {
                ExampleMessage message = NatsService.DeserializeMessage<ExampleMessage>(e.Message);
                Debug.Log($"Received message: {message.Content} (ID: {message.Id})");
                
                // Process the message here...
            });
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"Error handling message: {ex.Message}");
        }
    }
} 