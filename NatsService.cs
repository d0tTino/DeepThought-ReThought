using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NATS.Client;
using UnityEngine;
using Newtonsoft.Json;

/// <summary>
/// Provides NATS messaging functionality for Unity applications.
/// Handles connections to NATS server and provides methods for publishing and subscribing.
/// </summary>
public class NatsService : MonoBehaviour
{
    [SerializeField] private string _natsServerUrl = "nats://localhost:4222";
    [SerializeField] private bool _connectOnStart = true;
    [SerializeField] private bool _useJetStream = true;
    
    private IConnection _connection;
    private Options _connectionOptions;
    private IJetStream _jetStream;
    private CancellationTokenSource _cts;
    
    // Event for connection status changes
    public event Action<bool> OnConnectionStatusChanged;
    
    /// <summary>
    /// Gets a value indicating whether the client is connected to the NATS server.
    /// </summary>
    public bool IsConnected => _connection?.State == ConnState.CONNECTED;

    private void Awake()
    {
        _cts = new CancellationTokenSource();
        SetupConnectionOptions();
    }
    
    private void Start()
    {
        if (_connectOnStart)
        {
            Connect();
        }
    }
    
    private void OnDestroy()
    {
        _cts.Cancel();
        Disconnect();
    }
    
    /// <summary>
    /// Configures the connection options for the NATS client.
    /// </summary>
    private void SetupConnectionOptions()
    {
        _connectionOptions = ConnectionFactory.GetDefaultOptions();
        _connectionOptions.Url = _natsServerUrl;
        
        // Setup event handlers
        _connectionOptions.DisconnectedEventHandler = (sender, args) => 
        {
            Debug.LogWarning($"Disconnected from NATS server: {args.Error}");
            OnConnectionStatusChanged?.Invoke(false);
        };
        
        _connectionOptions.ReconnectedEventHandler = (sender, args) => 
        {
            Debug.Log("Reconnected to NATS server");
            OnConnectionStatusChanged?.Invoke(true);
        };
        
        _connectionOptions.ClosedEventHandler = (sender, args) => 
        {
            Debug.Log("NATS connection closed");
            OnConnectionStatusChanged?.Invoke(false);
        };
    }
    
    /// <summary>
    /// Connects to the NATS server.
    /// </summary>
    /// <returns>True if connection successful, otherwise false.</returns>
    public bool Connect()
    {
        try
        {
            if (IsConnected)
            {
                Debug.LogWarning("Already connected to NATS server");
                return true;
            }
            
            _connection = new ConnectionFactory().CreateConnection(_connectionOptions);
            Debug.Log($"Connected to NATS server at {_natsServerUrl}");
            
            if (_useJetStream)
            {
                _jetStream = _connection.CreateJetStreamContext();
            }
            
            OnConnectionStatusChanged?.Invoke(true);
            return true;
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to connect to NATS server: {ex.Message}");
            OnConnectionStatusChanged?.Invoke(false);
            return false;
        }
    }
    
    /// <summary>
    /// Disconnects from the NATS server.
    /// </summary>
    public void Disconnect()
    {
        if (_connection != null)
        {
            try
            {
                _connection.Drain();
                _connection.Close();
                _connection = null;
                _jetStream = null;
                Debug.Log("Disconnected from NATS server");
                OnConnectionStatusChanged?.Invoke(false);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error disconnecting from NATS server: {ex.Message}");
            }
        }
    }
    
    /// <summary>
    /// Publishes a message to the specified subject.
    /// </summary>
    /// <param name="subject">The subject to publish to.</param>
    /// <param name="data">The data to publish.</param>
    /// <typeparam name="T">Type of data to publish.</typeparam>
    /// <returns>True if publish successful, otherwise false.</returns>
    public bool Publish<T>(string subject, T data)
    {
        if (!IsConnected)
        {
            Debug.LogError("Cannot publish: Not connected to NATS server");
            return false;
        }
        
        try
        {
            string jsonData = JsonConvert.SerializeObject(data);
            byte[] messageData = Encoding.UTF8.GetBytes(jsonData);
            
            if (_useJetStream && _jetStream != null)
            {
                _jetStream.Publish(subject, messageData);
            }
            else
            {
                _connection.Publish(subject, messageData);
            }
            
            return true;
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error publishing message to {subject}: {ex.Message}");
            return false;
        }
    }
    
    /// <summary>
    /// Subscribes to the specified subject and invokes the handler when messages are received.
    /// </summary>
    /// <param name="subject">The subject to subscribe to.</param>
    /// <param name="handler">The event handler for received messages.</param>
    /// <returns>The subscription or null if subscription failed.</returns>
    public IAsyncSubscription Subscribe(string subject, EventHandler<MsgHandlerEventArgs> handler)
    {
        if (!IsConnected)
        {
            Debug.LogError("Cannot subscribe: Not connected to NATS server");
            return null;
        }
        
        try
        {
            IAsyncSubscription subscription = _connection.SubscribeAsync(subject);
            subscription.MessageHandler += handler;
            subscription.Start();
            Debug.Log($"Subscribed to {subject}");
            return subscription;
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error subscribing to {subject}: {ex.Message}");
            return null;
        }
    }
    
    /// <summary>
    /// Deserializes a received message to the specified type.
    /// </summary>
    /// <param name="msg">The received message.</param>
    /// <typeparam name="T">Type to deserialize to.</typeparam>
    /// <returns>The deserialized object.</returns>
    public static T DeserializeMessage<T>(Msg msg)
    {
        try
        {
            string jsonData = Encoding.UTF8.GetString(msg.Data);
            return JsonConvert.DeserializeObject<T>(jsonData);
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error deserializing message: {ex.Message}");
            return default;
        }
    }
} 