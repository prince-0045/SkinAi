# STATE MANAGEMENT

Complete documentation of how state is managed in the SkinAI application. Currently uses React Context API (no Redux, Zustand, or MobX).

---

## Overview

**State Management Pattern:** React Context API + localStorage

**Global State:** Authentication context (user, token)

**Local State:** Component-level state using `useState()`

**No Redux/Zustand:** Application is relatively simple, Context API sufficient

---

## Global State: AuthContext

### Location
```
frontend/src/context/AuthContext.jsx
```

### Purpose
Manages all user authentication state and related operations throughout the application.

### State Variables

#### `user` (Object | null)
```javascript
user = {
  name: "John Doe",        // User's full name
  email: "john@example.com", // User's email (unique identifier)
  created_at: "2025-03-31T12:00:00" // Account creation date
}
```

**Default:** `null` (not logged in)

**Updated by:**
- `login()` — Sets user after successful login
- `signup()` — Sets user after successful signup
- `googleLogin()` — Sets user after Google auth
- `logout()` — Clears to null

**Used by:**
- Navbar (display user name/email)
- Profile.jsx (show user info)
- All protected routes (verify user exists)
- Admin (show welcome message)

#### `loading` (Boolean)
```javascript
loading = true  // During initialization
loading = false // After localStorage restored
```

**Default:** `true`

**Updated by:**
- `initAuth()` useEffect — Sets false after checking localStorage

**Used by:**
- App.jsx ProtectedRoute component (show loading while initializing)
- Pages use it to conditionally render

#### `token` (String, not in state)
**Note:** Token is NOT stored in React state, only in localStorage
- Key: `localStorage.token`
- Value: JWT string
- Persists across page refreshes
- Used in Authorization header for API calls

### Methods

#### `login(email, password)`
```javascript
const result = await login("john@example.com", "Password123");
// result = {success: true} or {success: false, error: "message"}
```

**What it does:**
1. Calls `/api/v1/auth/login` with email/password
2. On success: Stores token + user in localStorage
3. Updates React state with user
4. On error: Returns error message

**Side effects:**
- localStorage.token updated
- localStorage.user updated
- User state updated
- HTTP request to backend

**Used by:** Login.jsx form

#### `signup(name, email, password)`
```javascript
const result = await signup("John", "john@example.com", "Password123");
// result = {success: true, user: {...}} or {success: false, error: "..."}
```

**What it does:**
1. Calls `/api/v1/auth/signup` with credentials
2. Backend auto-generates token (direct login)
3. Stores token + user in localStorage
4. Updates state

**Side effects:** Same as login()

**Used by:** Signup.jsx form

#### `googleLogin(token)`
```javascript
const accessToken = tokenResponse.access_token; // From Google OAuth
const result = await googleLogin(accessToken);
```

**What it does:**
1. Calls `/api/v1/auth/google` with Google access token
2. Backend validates with Google API
3. Creates/finds user by email
4. Returns JWT

**Side effects:** localStorage + state updates

**Used by:** Login.jsx and Signup.jsx with `@react-oauth/google`

#### `logout()`
```javascript
logout(); // User clicks logout button
```

**What it does:**
1. Clears localStorage.token
2. Clears localStorage.user
3. Sets user state to null

**Side effects:**
- localStorage cleared
- State updated

**Used by:** Navbar logout button, Profile.jsx

#### `verifyOtp(email, otp)`
```javascript
const result = await verifyOtp("john@example.com", "123456");
```

**What it does:**
1. Calls `/api/v1/auth/verify-otp` (NOT IMPLEMENTED)
2. Returns success/error

**Status:** [INCOMPLETE] - Endpoint exists but frontend integration not finished

**Used by:** OTP.jsx (if completed)

---

## Context Consumer Hook

### `useAuth()`

**Usage:**
```javascript
import { useAuth } from '../context/AuthContext';

function MyComponent() {
  const { user, loading, login, logout, signup } = useAuth();
  
  if (loading) return <div>Loading...</div>;
  if (!user) return <div>Not logged in</div>;
  
  return <div>Welcome {user.name}!</div>;
}
```

**Returns:**
```javascript
{
  user: {name, email, created_at} | null,
  loading: boolean,
  login: async function(email, password),
  signup: async function(name, email, password),
  googleLogin: async function(token),
  logout: function(),
  verifyOtp: async function(email, otp)
}
```

**Used by:** Nearly every page and component

---

## Storage: localStorage

### Stored Data

#### `localStorage.token` (String)
```javascript
localStorage.token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Purpose:** JWT token for authenticated requests

**Persistence:** Survives page refresh + browser close

**Accessed by:** fetch() calls with Authorization header

**Cleared by:** `logout()`

#### `localStorage.user` (JSON String)
```javascript
localStorage.user = JSON.stringify({
  name: "John Doe",
  email: "john@example.com",
  created_at: "2025-03-31T12:00:00"
})
```

**Purpose:** Cache user info for faster page loads

**Persistence:** Survives page refresh

**Accessed by:** AuthContext on app load

**Cleared by:** `logout()`

---

## Initialization Flow

```
1. App mounts
   ↓
2. AuthProvider component renders
   ↓
3. useEffect hook runs (once on mount)
   ↓
4. Check localStorage for token + user
   ↓
5. If both exist:
   └─ Parse user JSON
   └─ Set user state to parsed object
   └─ Loading remains true for now
   ↓
6. Set loading = false
   ↓
7. Render children (pages can now use useAuth)
```

---

## Protected Route Implementation

```javascript
// App.jsx
const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) return <div>Loading...</div>;

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return children;
};

// Usage:
<Route path="/detect" element={
  <ProtectedRoute>
    <Detect />
  </ProtectedRoute>
} />
```

**Flow:**
1. Route component mounts
2. useAuth() hook gets user + loading
3. If loading: Show spinner
4. If !user: Redirect to /login
5. Else: Render protected page

---

## Public Route Implementation

```javascript
// App.jsx
const PublicRoute = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) return <div>Loading...</div>;

  if (user) {
    return <Navigate to="/track" replace />;
  }

  return children;
};

// Usage:
<Route path="/login" element={
  <PublicRoute>
    <Login />
  </PublicRoute>
} />
```

**Flow:**
1. If user is logged in: Redirect to /track
2. If user not logged in: Show login page

---

## Automatic Heartbeat (Pulse)

**Purpose:** Track user's real-time activity for admin dashboard

**Implementation:**
```javascript
useEffect(() => {
  let interval;
  if (user && !loading) {
    const sendPulse = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) return;
        
        await fetch(`${API_BASE_URL}/api/v1/users/pulse`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
      } catch (err) {
        // Silently ignore heartbeat failures
      }
    };

    sendPulse(); // Initial pulse
    interval = setInterval(sendPulse, 60000); // Every 60 seconds
  }
  return () => clearInterval(interval);
}, [user, loading, API_BASE_URL]);
```

**Frequency:** Every 60 seconds (1 minute)

**What it does:** Updates ActiveSession.last_seen_at in backend MongoDB

**Error handling:** Silently ignores failures (doesn't affect user experience)

---

## Local Component State

### By Page

#### `Login.jsx`
```javascript
const [email, setEmail] = useState('');
const [password, setPassword] = useState('');
const [showPassword, setShowPassword] = useState(false);
const [error, setError] = useState('');
const [loading, setLoading] = useState(false);
```

**Purpose:** Form inputs, UI feedback, loading state

#### `Signup.jsx`
```javascript
const [email, setEmail] = useState('');
const [name, setName] = useState('');
const [password, setPassword] = useState('');
const [showPassword, setShowPassword] = useState(false);
const [error, setError] = useState('');
const [loading, setLoading] = useState(false);
```

#### `Detect.jsx`
```javascript
const [file, setFile] = useState(null);
const [preview, setPreview] = useState(null);
const [isAnalyzing, setIsAnalyzing] = useState(false);
const [result, setResult] = useState(null);
const [error, setError] = useState(null);
const [uploadLimit, setUploadLimit] = useState(null);
```

**Purpose:**
- `file` — Selected image for analysis
- `preview` — Image preview URL for display
- `isAnalyzing` — Show/hide loading spinner during inference
- `result` — ML prediction result
- `error` — Error message
- `uploadLimit` — Remaining daily scans

#### `Tracker.jsx`
```javascript
const [history, setHistory] = useState([]);
const [loading, setLoading] = useState(true);
const [error, setError] = useState(null);
const [viewMode, setViewMode] = useState('grid'); // 'grid' or 'chart'
```

**Purpose:**
- `history` — All user's past scans
- `loading` — API request status
- `error` — Fetch error message
- `viewMode` — Toggle grid/chart view

#### `Profile.jsx`
```javascript
const [activeTab, setActiveTab] = useState('history');
const [history, setHistory] = useState([]);
const [loading, setLoading] = useState(true);
const [isChangePasswordOpen, setIsChangePasswordOpen] = useState(false);
const [changePasswordForm, setChangePasswordForm] = useState({
  currentPassword: '',
  newPassword: '',
  confirmPassword: ''
});
const [changePasswordStatus, setChangePasswordStatus] = useState({ 
  type: '', 
  message: '' 
});
const [showPasswords, setShowPasswords] = useState(false);
```

**Purpose:**
- Account settings UI state
- Form inputs for password change
- Tab selection (history, settings, etc.)

#### `Doctors.jsx`
```javascript
// No local state - DoctorFinder component handles it
```

#### `Admin.jsx`
```javascript
// Fetches stats from /api/v1/admin/stats
// Stores in component state
```

---

## Data Flow with State

### Scan Detection Flow

```
Detect.jsx
  ├── file: null → User selects image → file: File object
  ├── preview: null → URL.createObjectURL → preview: blob URL
  ├── isAnalyzing: false → User clicks Analyze → isAnalyzing: true
  │   └── Frontend compresses image
  │   └── Calls /api/v1/scan/predict
  │       └── Backend: ML inference (slow)
  │   ├── Response received
  │   ├── isAnalyzing: true → false
  │   ├── result: null → result: {disease, confidence, ...}
  │   └── UI re-renders with results
  └── User clicks Download PDF
      └── Calls downloadMedicalReport(result, user)

Tracker.jsx  
  ├── history: []
  ├── loading: true
  ├── Fetch /api/v1/scan/history
  ├── Response: Array of scans
  ├── history: [] → history: [scan1, scan2, ...]
  ├── loading: true → false
  └── Render history grid or chart
```

---

## State Coupling & Dependencies

### Tightly Coupled (Depends on each other)

```
user state ←→ token (localStorage)
  - They must stay in sync
  - Logout clears both
  - Login sets both
  - Initialization loads both
```

### Loosely Coupled (Independent)

```
Page-level state (Detect.jsx, Tracker.jsx, etc.)
  - Not connected to AuthContext
  - Can be modified independently
  - Don't affect other pages
```

---

## State Persistence Strategy

| State | Storage | Persistence | Expiry |
|---|---|---|---|
| user | localStorage | Survives refresh | Never (manual logout) |
| token | localStorage | Survives refresh | 24 hours (JWT exp) |
| file (upload) | Memory | Lost on refresh | Page navigation |
| result (scan) | Memory | Lost on refresh | User leaves page |
| history (scans) | Memory | Lost on refresh | useEffect re-fetches |

---

## Common State Patterns

### Loading State Pattern

```javascript
const [loading, setLoading] = useState(true);
const [error, setError] = useState(null);

useEffect(() => {
  const fetch = async () => {
    try {
      const data = await API.get(...);
      setData(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };
  fetch();
}, []);

// In render:
if (loading) return <Spinner />;
if (error) return <Error message={error} />;
return <Content data={data} />;
```

### Form State Pattern

```javascript
const [form, setForm] = useState({
  email: '',
  password: ''
});

const handleChange = (e) => {
  const { name, value } = e.target;
  setForm(prev => ({
    ...prev,
    [name]: value
  }));
};

const handleSubmit = async (e) => {
  e.preventDefault();
  const result = await login(form.email, form.password);
  // Handle result
};
```

### Toggle State Pattern

```javascript
const [showPassword, setShowPassword] = useState(false);

const toggle = () => setShowPassword(!showPassword);

<input type={showPassword ? "text" : "password"} />
<button onClick={toggle}>Show/Hide</button>
```

### Multi-Tab State Pattern

```javascript
const [activeTab, setActiveTab] = useState('history');

<button onClick={() => setActiveTab('history')}>History</button>
<button onClick={() => setActiveTab('settings')}>Settings</button>

{activeTab === 'history' && <History />}
{activeTab === 'settings' && <Settings />}
```

---

## State Debugging

### To Check Current Auth State

Open developer console and run:
```javascript
console.log(localStorage.getItem('token'));
console.log(JSON.parse(localStorage.getItem('user')));
```

### To Clear All State

```javascript
localStorage.clear();
location.reload();
```

### To Test Protected Routes

1. Logout (clears user)
2. Try navigating to `/detect` → Should redirect to `/login`
3. Login → Should redirect to `/detect`

---

## State Limitations & Issues

### Known Issues

1. **No Redux/Advanced State Management**
   - Impact: For complex apps, Context can cause unnecessary re-renders
   - Current: Works fine for this app size
   
2. **State Only in AuthContext**
   - Impact: Scan history, upload results stored locally (not global)
   - Reason: Simpler,page-level management sufficient

3. **Token Expiry Not Handled**
   - Impact: If token expires mid-session, API calls will fail with 401
   - Current: User can re-login, token gets renewed
   
4. **OTP Flow Incomplete**
   - Impact: verifyOtp() function exists but unused
   - Status: To be completed

### Potential Improvements

- Add Redux for larger app
- Implement token refresh (before expiry)
- Add optimistic updates for scans
- Implement undo/redo for form inputs
- Add persistence for draft uploads

---

## Summary

**State Management Strategy:**
- Global: React Context (auth only)
- Local: Component useState (page features)
- Persistent: localStorage (token + user)
- Backend sync: API calls (REST)

**Philosophy:** Keep it simple, use React Context for auth, component state for features

**Scalability:** Works well for current app size. Upgrade to Redux if app grows significantly.

