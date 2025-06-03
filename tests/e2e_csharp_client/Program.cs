using System;
using System.Text;
using System.Threading;
using NATS.Client;

public class E2ETestClient
{
    public static int Main(string[] args)
    {
        if (args.Length < 5)
        {
            Console.Error.WriteLine("Usage: E2ETestClient <natsUrl> <publishSubject> <messagePayload> <subscribeSubject> <timeoutMilliseconds>");
            return 1; // Argument error
        }

        string natsUrl = args[0];
        string publishSubject = args[1];
        string messagePayload = args[2];
        string subscribeSubject = args[3];
        if (!int.TryParse(args[4], out int timeoutMilliseconds))
        {
            Console.Error.WriteLine("Invalid timeout value.");
            return 1;
        }

        var receivedMessageSignal = new ManualResetEventSlim(false);
        string? receivedMessage = null;

        Options opts = ConnectionFactory.GetDefaultOptions();
        opts.Url = natsUrl;
        opts.Timeout = 5000; // Connection timeout

        IConnection? nc = null;
        IAsyncSubscription? sub = null;

        try
        {
            Console.Out.WriteLine($"Attempting to connect to NATS at {natsUrl}...");
            nc = new ConnectionFactory().CreateConnection(opts);
            Console.Out.WriteLine("Connected to NATS.");

            sub = nc.SubscribeAsync(subscribeSubject, (sender, eventArgs) =>
            {
                receivedMessage = Encoding.UTF8.GetString(eventArgs.Message.Data);
                Console.Out.WriteLine($"Received message on '{subscribeSubject}': {receivedMessage}");
                receivedMessageSignal.Set();
            });
            nc.Flush(5000); // Ensure subscription is processed
            Console.Out.WriteLine($"Subscribed to '{subscribeSubject}'.");

            Console.Out.WriteLine($"Publishing '{messagePayload}' to '{publishSubject}'...");
            nc.Publish(publishSubject, Encoding.UTF8.GetBytes(messagePayload));
            nc.Flush(5000); // Ensure message is sent
            Console.Out.WriteLine("Message published.");

            Console.Out.WriteLine($"Waiting for response for {timeoutMilliseconds}ms...");
            if (receivedMessageSignal.Wait(timeoutMilliseconds))
            {
                Console.Out.WriteLine($"Success: Response received.");
                Console.Out.WriteLine(receivedMessage);
                return 0; // Success
            }
            else
            {
                Console.Error.WriteLine("Error: Timeout waiting for response message.");
                return 2; // Timeout error
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.GetType().Name} - {ex.Message}");
            if(ex.StackTrace != null) Console.Error.WriteLine(ex.StackTrace.Split(new[] { Environment.NewLine }, StringSplitOptions.None)[0]); // First line of stack trace
            return 3; // NATS or other error
        }
        finally
        {
            if (sub != null && sub.IsValid)
            {
                try { sub.Unsubscribe(); } catch { /* ignore */ }
            }
            if (nc != null && nc.State == ConnState.CONNECTED)
            {
                try { nc.Drain(2000); nc.Close(); Console.Out.WriteLine("NATS connection closed."); } catch { /* ignore */ }
            }
        }
    }
}
