# 🌊 Function Flooding

Function Flooding attacks exploit AI agents' recursive planning capabilities to generate excessive tool calls, leading to resource exhaustion, financial damage, and system overload. A single malicious prompt can trigger exponential tool invocation patterns that overwhelm infrastructure and budgets.

## 🎯 Attack Concept

```mermaid
graph TD
    A[Single User Prompt] --> B[Agent Planning]
    B --> C[Tool Call 1]
    B --> D[Tool Call 2]
    B --> E[Tool Call N]
    
    C --> F[Sub-task 1.1]
    C --> G[Sub-task 1.2]
    D --> H[Sub-task 2.1]
    D --> I[Sub-task 2.2]
    E --> J[Sub-task N.1]
    E --> K[Sub-task N.2]
    
    F --> L[More Tool Calls...]
    G --> M[More Tool Calls...]
    
    style A fill:#FFB366
    style L fill:#FF6B6B
    style M fill:#FF6B6B
    
    classDef normal fill:#e6f3ff,stroke:#0066cc
    classDef dangerous fill:#ffe6e6,stroke:#cc0000
    
    class A,B,C,D,E,F,G,H,I,J,K normal
    class L,M dangerous
```

## 🔄 Attack Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as AI Agent
    participant T as Tool System
    participant S as Server/API
    participant B as Billing System
    
    U->>A: "Summarize 100 emails with 5 summaries each"
    A->>A: Plan: 100 × 5 = 500 tool calls
    
    loop 500 times
        A->>T: Call summarization tool
        T->>S: API request
        S->>B: Charge for API usage
        S->>T: Response
        T->>A: Summary result
    end
    
    Note over B: $$$$ Massive bill accumulation
    Note over S: Server overload
    A->>U: "Task completed"
```

## 🧠 Attack Variants

### 1. Exponential Fan-Out
Prompts that create multiplicative task expansion:

```
"Create 10 reports, each analyzing 20 data sources, with 5 different visualization types"
Result: 10 × 20 × 5 = 1,000 tool calls
```

### 2. Recursive Depth Attacks
Tasks that create nested sub-tasks:

```
"Analyze this document by breaking it into sections, then paragraphs, then sentences, then words"
Result: Exponential recursive breakdown
```

### 3. Iterative Refinement Loops
Prompts that trigger endless improvement cycles:

```
"Keep improving this code until it's perfect, testing each iteration"
Result: Infinite loop of code generation and testing
```

### 4. Cross-Product Combinations
Tasks requiring all possible combinations:

```
"Test every configuration option against every environment"
Result: N × M tool calls for N options and M environments
```

### 5. Parallel Processing Abuse
Exploit concurrent execution capabilities:

```
"Process these 1000 files simultaneously for maximum speed"
Result: 1000 concurrent tool calls overwhelming system
```

## 💥 Impact Assessment

### Financial Impact
- **API Cost Explosion**: Thousands of unexpected API calls
- **Resource Consumption**: Excessive compute and bandwidth usage
- **Budget Exhaustion**: Rapid depletion of allocated resources
- **Billing Shock**: Unexpected charges in the thousands

### Technical Impact
- **Server Overload**: Infrastructure unable to handle request volume
- **Rate Limiting**: APIs hitting usage limits and failing
- **Memory Exhaustion**: Agent systems running out of resources
- **Cascading Failures**: Overload causing system-wide issues

### Operational Impact
- **Service Degradation**: Legitimate users experiencing slowdowns
- **System Downtime**: Complete service failures under load
- **Support Burden**: Increased incident response and troubleshooting
- **Reputation Damage**: Poor user experience and reliability issues

## 🔍 Detection Strategies

### Real-Time Monitoring
```python
class FloodingDetector:
    """Detect function flooding attacks in real-time."""
    
    def __init__(self, max_calls_per_minute=100, max_depth=10):
        self.max_calls_per_minute = max_calls_per_minute
        self.max_depth = max_depth
        self.call_history = []
        self.recursion_depth = 0
    
    def check_flooding(self, tool_call: str) -> bool:
        """Check if current call pattern indicates flooding."""
        current_time = time.time()
        
        # Remove old calls (older than 1 minute)
        self.call_history = [
            call for call in self.call_history
            if current_time - call['timestamp'] < 60
        ]
        
        # Add current call
        self.call_history.append({
            'tool': tool_call,
            'timestamp': current_time
        })
        
        # Check rate limit
        if len(self.call_history) > self.max_calls_per_minute:
            return True
        
        # Check recursion depth
        if self.recursion_depth > self.max_depth:
            return True
        
        return False
```

### Pattern Analysis
```mermaid
graph TD
    A[Tool Call] --> B[Rate Analysis]
    A --> C[Pattern Detection]
    A --> D[Recursion Check]
    
    B --> E[Calls per Minute]
    C --> F[Repetitive Patterns]
    D --> G[Depth Tracking]
    
    E --> H{Threshold Exceeded?}
    F --> H
    G --> H
    
    H -->|Yes| I[Block & Alert]
    H -->|No| J[Allow Call]
    
    style I fill:#FF6B6B,color:#fff
```

### Cost Monitoring
```python
class CostMonitor:
    """Monitor API costs and detect unusual spending."""
    
    def __init__(self, daily_budget=100, alert_threshold=0.8):
        self.daily_budget = daily_budget
        self.alert_threshold = alert_threshold
        self.daily_spend = 0
        self.last_reset = time.time()
    
    def track_api_call(self, cost: float) -> bool:
        """Track API call cost and check budget."""
        # Reset daily counter if needed
        if time.time() - self.last_reset > 86400:  # 24 hours
            self.daily_spend = 0
            self.last_reset = time.time()
        
        self.daily_spend += cost
        
        # Check if approaching budget limit
        if self.daily_spend > self.daily_budget * self.alert_threshold:
            return False  # Block further calls
        
        return True
```

## 🛡️ Defense Mechanisms

### 1. Rate Limiting
```python
class ToolRateLimiter:
    """Implement rate limiting for tool calls."""
    
    def __init__(self, max_calls_per_minute=50, max_calls_per_hour=1000):
        self.max_calls_per_minute = max_calls_per_minute
        self.max_calls_per_hour = max_calls_per_hour
        self.minute_calls = []
        self.hour_calls = []
    
    def allow_call(self) -> bool:
        """Check if call is allowed under rate limits."""
        current_time = time.time()
        
        # Clean old calls
        self.minute_calls = [t for t in self.minute_calls if current_time - t < 60]
        self.hour_calls = [t for t in self.hour_calls if current_time - t < 3600]
        
        # Check limits
        if len(self.minute_calls) >= self.max_calls_per_minute:
            return False
        if len(self.hour_calls) >= self.max_calls_per_hour:
            return False
        
        # Record call
        self.minute_calls.append(current_time)
        self.hour_calls.append(current_time)
        return True
```

### 2. Budget Controls
```python
class BudgetController:
    """Control spending with hard budget limits."""
    
    def __init__(self, daily_limit=100, monthly_limit=2000):
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.daily_spend = 0
        self.monthly_spend = 0
    
    def authorize_spend(self, estimated_cost: float) -> bool:
        """Authorize spending based on budget limits."""
        if self.daily_spend + estimated_cost > self.daily_limit:
            return False
        if self.monthly_spend + estimated_cost > self.monthly_limit:
            return False
        return True
```

### 3. Complexity Analysis
```python
def analyze_prompt_complexity(prompt: str) -> dict:
    """Analyze prompt for potential flooding indicators."""
    complexity_indicators = {
        'multiplicative_keywords': 0,
        'large_numbers': 0,
        'recursive_patterns': 0,
        'combination_words': 0
    }
    
    # Check for multiplicative language
    multiplicative_patterns = [
        r'\d+.*each.*\d+',
        r'every.*\d+.*with.*\d+',
        r'\d+.*times.*\d+'
    ]
    
    for pattern in multiplicative_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            complexity_indicators['multiplicative_keywords'] += 1
    
    # Check for large numbers
    numbers = re.findall(r'\d+', prompt)
    for num in numbers:
        if int(num) > 100:
            complexity_indicators['large_numbers'] += 1
    
    return complexity_indicators
```

### 4. Circuit Breakers
```python
class CircuitBreaker:
    """Implement circuit breaker pattern for tool calls."""
    
    def __init__(self, failure_threshold=10, timeout=300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call_allowed(self) -> bool:
        """Check if calls are allowed based on circuit state."""
        current_time = time.time()
        
        if self.state == 'OPEN':
            if current_time - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
                return True
            return False
        
        return True
    
    def record_success(self):
        """Record successful call."""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def record_failure(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
```

## ▶️ Usage

```bash
export OPENAI_API_KEY=sk-...
python function_flooding_agent.py
```

## 🔬 Research Applications

### Red Team Testing
- Test organization's rate limiting effectiveness
- Evaluate budget control mechanisms
- Assess system resilience under load

### Blue Team Defense
- Develop flooding detection algorithms
- Create cost monitoring systems
- Build circuit breaker implementations

## 📊 Success Metrics

- **Call Amplification Factor**: Ratio of actual calls to expected calls
- **Cost Impact**: Financial damage from excessive API usage
- **System Degradation**: Performance impact on infrastructure
- **Detection Time**: How quickly flooding is identified and stopped

## ⚠️ Ethical Considerations

This attack can cause significant financial damage and service disruption. Use only for:
- Authorized penetration testing
- System resilience evaluation
- Defense mechanism development
- Security awareness training

Never deploy against systems without explicit written authorization.

## 🔗 Related Attacks

- **DDoS Attacks**: Similar resource exhaustion patterns
- **Fork Bombs**: Comparable exponential resource consumption
- **Algorithmic Complexity Attacks**: Related computational exploitation
- **Resource Exhaustion**: Similar infrastructure targeting