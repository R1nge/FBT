using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace _Assets.Scripts
{
    public class UdpListener : MonoBehaviour
    {
        private void Start()
        {
            Init();
        }

        public void Init()
        {
            Task.Run(Listen);
        }

        public void Listen()
        {
            byte[] data = new byte[1024];
            IPEndPoint ipep = new IPEndPoint(IPAddress.Any, 9050);
            UdpClient newsock = new UdpClient(ipep);
            
            Debug.Log($"Waiting for a client on {ipep.Address}:{ipep.Port}");

            IPEndPoint sender = new IPEndPoint(IPAddress.Any, 0);

            data = newsock.Receive(ref sender);

            Debug.Log($"Message received from {sender}");
            Debug.Log(Encoding.ASCII.GetString(data, 0, data.Length));

            string welcome = "Welcome to my test server";
            data = Encoding.ASCII.GetBytes(welcome);
            newsock.Send(data, data.Length, sender);

            while(true)
            {
                data = newsock.Receive(ref sender);

                Debug.Log(Encoding.ASCII.GetString(data, 0, data.Length));
                newsock.Send(data, data.Length, sender);
            }
        }
    }
}