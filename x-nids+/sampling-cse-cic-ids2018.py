from dotenv import load_dotenv
from os import getenv
from datasets import load_dataset
import pandas as pd
import csv
from tqdm import tqdm


load_dotenv()
HUGGING_FACE_READ_TOKEN = getenv("HUGGING_FACE_READ_TOKEN")

index = 0
dataset = load_dataset("Jetlime/NF-CSE-CIC-IDS2018-v2", streaming=False, split="test")
dataset = dataset.train_test_split(test_size=0.000885, seed=1234, stratify_by_column="Attack")
dataset = dataset["test"]

index = 0
dataset_full = load_dataset("Jetlime/NF-CSE-CIC-IDS2018-v2", streaming=False, split="test")
dataset_full = dataset_full.to_pandas()

def string_to_dict(data_str):
    # Split the string by commas
    items = data_str.split(', ')
    
    # Initialize an empty dictionary
    result = {}

    for item in items:
        # Split each item by the colon to separate key and value
        key, value = item.split(': ')

        # Remove whitespace and handle numeric values
        key = key.strip()
        
        # Convert value to appropriate type
        if value.isdigit():
            result[key] = int(value)
        elif value.replace('.', '', 1).isdigit():  # Check for floats
            result[key] = float(value)
        else:
            result[key] = value.strip()  # String case
    
    return result

def excel_to_dict(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Convert the DataFrame to a dictionary
    data_dict = df.to_dict(orient='records')
    
    return data_dict

def find_previous_connections(netflow, src_ip, dst_ip):
    row_index = dataset_full.loc[dataset_full['input'] == netflow].index[0]
    filtered_rows = dataset_full[(dataset_full['input'].str.contains(src_ip)) | (dataset_full['input'].str.contains(dst_ip))]
    return filtered_rows[filtered_rows.index < row_index].tail(3)

def csv_to_dict(file_path):
    # Read the Excel file
    df = pd.read_csv(file_path)
    # Convert the DataFrame to a dictionary
    data_dict = df.to_dict(orient='records')
    return data_dict

def print_table(table):
    texts = []
    for index, row in table.iterrows():
        texts.append(row['input'])
        # texts.append(entry)
    return "\n".join(texts)

file_path = './l7_proto.xlsx'
l7_proto_list = excel_to_dict(file_path)
file_path = './protocol.csv'
protocol_list = csv_to_dict(file_path)
ip_intelligence = csv_dict = [row for row in csv.DictReader(open("./ip_information.csv"))]
all_ips = []
for netflow_sample in tqdm(dataset):
    if netflow_sample["output"] == 1:
        netflow_dict = string_to_dict(netflow_sample['input'])
        
        src_ip = netflow_dict["IPV4_SRC_ADDR"]
        dst_ip = netflow_dict["IPV4_DST_ADDR"]
        ip_prev = find_previous_connections(netflow_sample['input'], src_ip, dst_ip)

        protocol = netflow_dict["PROTOCOL"]
        l7_proto = netflow_dict["L7_PROTO"]
        l7_proto_details = l7_proto_list[int(l7_proto)]
        protocol_details = protocol_list[int(protocol)]

        ip_intelligence_src = [entry for entry in ip_intelligence if entry['ip'] == src_ip][0]
        ip_intelligence_dst = [entry for entry in ip_intelligence if entry['ip'] == dst_ip][0]
        index+=1
        prompt = f"""The Network Intrusion Detection System has flagged the following NetFlow data as malicious. Provide an explanation detailing why it is considered malicious, citing specific feature values present in the NetFlow sample to support your analysis

Each netflow entry will be provided as key-value pairs representing the following features:

IPV4 SRC ADDR IPv4 source address IPV4 DST ADDR IPv4 destination address L4 SRC PORT IPv4 source port number L4 DST PORT IPv4 destination port number PROTOCOL IP protocol identifier byte L7 PROTO Application protocol (numeric) IN BYTES Incoming number of bytes OUT BYTES Outgoing number of bytes IN PKTS Incoming number of packets OUT PKTS Outgoing number of packets FLOW DURATION MILLISECONDS Flow duration in milliseconds TCP FLAGS Cumulative of all TCP flags CLIENT TCP FLAGS Cumulative of all client TCP flags SERVER TCP FLAGS Cumulative of all server TCP flags DURATION IN Client to Server stream duration (msec) DURATION OUT Client to Server stream duration (msec) MIN TTL Min flow TTL MAX TTL Max flow TTL LONGEST FLOW PKT Longest packet (bytes) of the flow SHORTEST FLOW PKT Shortest packet (bytes) of the flow MIN IP PKT LEN Len of the smallest flow IP packet observed MAX IP PKT LEN Len of the largest flow IP packet observed SRC TO DST SECOND BYTES Src to dst Bytes/sec DST TO SRC SECOND BYTES Dst to src Bytes/sec RETRANSMITTED IN BYTES Number of retransmitted TCP flow bytes (src->dst) RETRANSMITTED IN PKTS Number of retransmitted TCP flow packets (src->dst) RETRANSMITTED OUT BYTES Number of retransmitted TCP flow bytes (dst->src) RETRANSMITTED OUT PKTS Number of retransmitted TCP flow packets (dst->src) SRC TO DST AV G THROUGHPUT Src to dst average thpt (bps) DST TO SRC AV G THROUGHPUT Dst to src average thpt (bps) NUM PKTS UP TO 128 BYTES Packets whose IP size <= 128 NUM PKTS 128 TO 256 BYTES Packets whose IP size > 128 and <= 256 NUM PKTS 256 TO 512 BYTES Packets whose IP size > 256 and <= 512 NUM PKTS 512 TO 1024 BYTES Packets whose IP size > 512 and <= 1024

Detecting malicious netflow involves identifying patterns and behaviors that deviate from normal network activity. Here are some common indicators of malicious netflow:

High Traffic Volume: Sudden spikes in traffic volume may indicate data exfiltration or a Distributed Denial of Service (DDoS) attack.
Low Traffic Volume: Stealthy low-and-slow attacks may generate low volumes of traffic to avoid detection.
Unexpected Protocols: Use of uncommon or non-standard protocols for specific network segments.
Port Scanning: A high number of connection attempts to various ports, indicative of port scanning activities.
Data Exfiltration: Large amounts of data sent to an external IP that is not commonly contacted.
Command and Control (C&C) Traffic: Regular, periodic communications with external IPs or domains associated with malware command and control servers.
Anomalous Packet Size: Deviations in average packet size, such as unusually large or small packets.
Unusual Packet Flags: Unusual combinations of TCP flags, indicative of scanning or other malicious activities.
High Number of Failed Connections: Numerous failed connection attempts may indicate brute-force attacks or reconnaissance activities.
Large Outbound Data Transfers: Unusually large volumes of outbound data, which may indicate data exfiltration attempts.
Persistent Connections: Long-lasting connections that deviate from normal session durations, potentially indicating an attacker maintaining access to a system.

The Layer 7 protocol {int(l7_proto)} corresponds to the {l7_proto_details["Protocol"]} protocol. {'A protocol based on ' + l7_proto_details["Service"] + ', usually used for ' +  l7_proto_details["Description"] + '.' if l7_proto_details["Protocol"] != 'Unknown' else ''}

The IP Layer protocol {int(protocol)} corresponds to {protocol_details["Protocol"]} ({protocol_details["Keyword"]}). 

Source IP {src_ip} {' originates from ' + ip_intelligence_src['location'] + ' and has been known for ' + ip_intelligence_src['intelligence'] if ip_intelligence_src['intelligence'] else 'is an internal IP address'}.

Destination IP {dst_ip} {' originates from ' + ip_intelligence_dst['location'] + ' and has been known for ' + ip_intelligence_dst['intelligence'] if ip_intelligence_dst['intelligence'] else 'is an internal IP address'}.

3 connections related to these IP addresses preceding the malicious netflow were found:

{print_table(ip_prev)}

The Network Intrusion Detection System has flagged the following NetFlow data as malicious. Provide an explanation detailing why it is considered malicious, citing specific feature values present in the NetFlow sample to support your analysis.

Malicious NetFlow:\n{netflow_sample['input']}"""
        with open(f"./evaluation-set-cse-cic-ids2018/malicious-{str(index)}", "w") as f:
            f.write(prompt)