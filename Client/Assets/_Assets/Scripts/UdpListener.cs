using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using AsyncIO;
using NetMQ;
using NetMQ.Sockets;
using UnityEditor.VersionControl;
using UnityEngine;
using Task = System.Threading.Tasks.Task;

namespace _Assets.Scripts
{
    public class UdpListener : MonoBehaviour
    {
        public event Action<string> OnMessageReceived;
    
        private string _host;
        private string _port;

        private void Start()
        {
            Init(); 
        }

        public void Init()
        {
            _host = "127.0.0.1";
            _port = "9085";
            GetMessage("Hello!");
        }

        public async void GetMessage(string text)
        {
            var result = Task.Run(() => RequestMessage(text));
            await result;
            OnMessageReceived?.Invoke(result.Result);
        }

        private string RequestMessage(object text)
        {
            var messageReceived = false;
            var message = "";
            ForceDotNet.Force();
            var timeout = new TimeSpan(0, 0, 30);
        
            using (var socket = new RequestSocket())
            {
                socket.Connect($"tcp://{_host}:{_port}");
                if (socket.TrySendFrame($"{text}"))
                {
                    messageReceived = socket.TryReceiveFrameString(timeout, out message);

                    if (messageReceived)
                    {
                        Debug.Log($"Socket has received a message: {message}");
                    }
                }
            }

            NetMQConfig.Cleanup();
            if (!messageReceived)
            {
                message = "Could not receive message from server!";
                Debug.LogWarning(message);
            }

            return message;
        }
    }
}