// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FedXChain
 * @dev Smart contract for federated learning with explainability and trust management
 */
contract FedXChain {
    
    struct NodeInfo {
        address nodeAddress;
        string nodeId;
        uint256 trustScore;
        uint256 totalContributions;
        bool isActive;
    }
    
    struct ModelUpdate {
        string nodeId;
        uint256 round;
        bytes32 modelHash;
        bytes32 shapHash;
        uint256 timestamp;
        uint256 accuracy;
        uint256 fidelityScore;
    }
    
    struct AggregationLog {
        uint256 round;
        bytes32 globalModelHash;
        bytes32 globalShapHash;
        uint256 participatingNodes;
        uint256 timestamp;
        string metadata;
    }
    
    mapping(address => NodeInfo) public nodes;
    mapping(uint256 => AggregationLog) public aggregationLogs;
    mapping(uint256 => mapping(string => ModelUpdate)) public roundUpdates;
    
    address public coordinator;
    uint256 public currentRound;
    uint256 public totalNodes;
    
    event NodeRegistered(address indexed nodeAddress, string nodeId);
    event ModelUpdateSubmitted(string indexed nodeId, uint256 round, bytes32 modelHash);
    event AggregationCompleted(uint256 round, bytes32 globalModelHash, uint256 participatingNodes);
    event TrustScoreUpdated(string indexed nodeId, uint256 newScore);
    
    modifier onlyCoordinator() {
        require(msg.sender == coordinator, "Only coordinator can call this");
        _;
    }
    
    modifier onlyActiveNode() {
        require(nodes[msg.sender].isActive, "Node is not active");
        _;
    }
    
    constructor() {
        coordinator = msg.sender;
        currentRound = 0;
    }
    
    function registerNode(address _nodeAddress, string memory _nodeId) external onlyCoordinator {
        require(!nodes[_nodeAddress].isActive, "Node already registered");
        
        nodes[_nodeAddress] = NodeInfo({
            nodeAddress: _nodeAddress,
            nodeId: _nodeId,
            trustScore: 100, // Initial trust score
            totalContributions: 0,
            isActive: true
        });
        
        totalNodes++;
        emit NodeRegistered(_nodeAddress, _nodeId);
    }
    
    function submitUpdate(
        string memory _nodeId,
        bytes32 _modelHash,
        bytes32 _shapHash,
        uint256 _accuracy,
        uint256 _fidelityScore
    ) external onlyActiveNode {
        require(bytes(nodes[msg.sender].nodeId).length > 0, "Node not registered");
        
        roundUpdates[currentRound][_nodeId] = ModelUpdate({
            nodeId: _nodeId,
            round: currentRound,
            modelHash: _modelHash,
            shapHash: _shapHash,
            timestamp: block.timestamp,
            accuracy: _accuracy,
            fidelityScore: _fidelityScore
        });
        
        nodes[msg.sender].totalContributions++;
        
        emit ModelUpdateSubmitted(_nodeId, currentRound, _modelHash);
    }
    
    function logAggregation(
        bytes32 _globalModelHash,
        bytes32 _globalShapHash,
        uint256 _participatingNodes,
        string memory _metadata
    ) external onlyCoordinator {
        aggregationLogs[currentRound] = AggregationLog({
            round: currentRound,
            globalModelHash: _globalModelHash,
            globalShapHash: _globalShapHash,
            participatingNodes: _participatingNodes,
            timestamp: block.timestamp,
            metadata: _metadata
        });
        
        currentRound++;
        emit AggregationCompleted(currentRound - 1, _globalModelHash, _participatingNodes);
    }
    
    function updateTrustScore(address _nodeAddress, uint256 _newScore) external onlyCoordinator {
        require(nodes[_nodeAddress].isActive, "Node not active");
        nodes[_nodeAddress].trustScore = _newScore;
        emit TrustScoreUpdated(nodes[_nodeAddress].nodeId, _newScore);
    }
    
    function getNodeInfo(address _nodeAddress) external view returns (NodeInfo memory) {
        return nodes[_nodeAddress];
    }
    
    function getModelUpdate(uint256 _round, string memory _nodeId) external view returns (ModelUpdate memory) {
        return roundUpdates[_round][_nodeId];
    }
    
    function getAggregationLog(uint256 _round) external view returns (AggregationLog memory) {
        return aggregationLogs[_round];
    }
}
