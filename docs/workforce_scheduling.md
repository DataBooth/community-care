<!-- NOTE: See Google Docs version for latest -->

# Workforce Scheduling in Care Services

## Overview

Workforce scheduling in care services (such as in-home aged care or disability support) is a complex optimisation problem. It must balance the needs and constraints of both carers and care recipients, including skills, availability, continuity of care, preferences, regulatory requirements, and travel times.

## Common Approaches

- **Mathematical Optimisation:** Linear programming (LP), integer programming (IP), and mixed-integer programming (MIP) are widely used to model and solve scheduling problems with constraints like shift coverage, maximum hours, and skill matching.
- **Constraint Programming (CP):** CP is effective for handling complex rules (e.g., legal regulations, carer preferences, continuity of care) and is often used in workforce scheduling tools.
- **Metaheuristics:** Methods such as genetic algorithms, simulated annealing, and tabu search are applied to large, real-world problems where exact methods are computationally expensive.
- **Multi-objective Optimisation:** Many studies focus on balancing multiple goals, such as minimising costs, maximising continuity of care, and improving staff satisfaction.


## Key Constraints and Rules

- Carer qualifications and skills (matching to recipient needs)
- Carer and recipient availability and preferences
- Regulatory requirements (rest breaks, maximum hours, etc.)
- Continuity of care (same carer for same recipient)
- Travel time and geographic constraints
- Emergency and last-minute changes


## Notable Research Papers \& Reviews

- A 2023 review, "Optimisation approaches for home care scheduling: A review," covers mathematical and heuristic approaches, focusing on constraints relevant to aged and disability care.
- "Home health care routing and scheduling: A review" (European Journal of Operational Research, 2015) surveys models and solution methods for home care, emphasising real-world constraints.
- "A multi-objective approach to staff scheduling in home care" (Omega, 2017) discusses balancing cost, continuity, and satisfaction.


## Vendor Systems - Open Source and Commercial

- Some open-source projects and commercial systems (e.g., Skedulo, Allocate, AlayaCare) implement these techniques, often using a combination of optimisation and manual overrides.

If you have specific constraints, care types, or system scales in mind, please clarify so I can recommend the most relevant models or papers.

---

### Open source?

Several open source workforce scheduling systems are available that can be adapted for care workforce scheduling:

- **Staffjoy**: Open-source, self-hosted scheduling software that supports compliance rules, shift management, time-off requests, and role-based assignments. While no longer actively developed, the code is freely available and can be customised[^2_5].
- **OptaWeb Employee Rostering (OptaPlanner)**: An advanced, open-source, constraint-solving platform that uses metaheuristics and AI for shift scheduling. It supports employee skills, preferences, and complex constraints, making it suitable for care scenarios[^2_5].
- **Cal.com**: Free and open-source scheduling software, primarily designed for appointments but can be adapted for workforce scheduling needs[^2_7].
- **Auto Shift Planner**: Open-source tool that automates employee shift schedules using heuristic and metaheuristic algorithms, suitable for organisations with complex scheduling needs[^2_2].

These platforms provide flexibility, transparency, and the ability to customise for specific care sector rules and constraints[^2_5][^2_6]. Most commercial systems mentioned (like OnShift, Rotageek, QGenda) are not open source[^2_1][^2_3].


[^2_1]: https://blog.makeshift.ca/scheduling-software-for-healthcare

[^2_2]: https://peoplemanagingpeople.com/tools/best-open-source-employee-scheduling-software/

[^2_3]: https://www.onshift.com/products/workforce-management-software/workforce-scheduling

[^2_4]: https://cal.com/blog/integrating-open-source-appointment-scheduling-in-healthcare

[^2_5]: https://www.timetackle.com/all-about-open-source-employee-scheduling-software-for-2024/

[^2_6]: https://www.myshyft.com/blog/open-source-scheduling-software/

[^2_7]: https://www.goodfirms.co/scheduling-software/blog/best-free-open-source-scheduling-software

[^2_8]: https://www.symplr.com/products/symplr-workforce


---

## Mathematical Formulation for Care Workforce Scheduling



# What about allowing for recipient preference for carers?

To incorporate recipient (client) preferences for carers in the mathematical formulation, you can introduce a **preference parameter** and adjust the objective function or add soft constraints:


# Are there any good review research papers that look at this problem?

Yes, several review papers examine workforce scheduling in healthcare and care services, including approaches that address complex constraints and preferences:

- **"Task scheduling in long-term care facilities: A client-centered approach"** (Lieder et al., 2015) reviews scheduling methods with a focus on client needs and preferences, relevant for aged care and disability services[^5_3].
- **"Centralised Scheduling of Nursing Staff: A Rapid Review of the Literature"** (Morse et al., 2024) summarises evidence on centralised scheduling models, their impact on productivity, staff satisfaction, and approaches used in hospital settings[^5_5].
- **"Development of a new personalised staff-scheduling method with a work-life balance perspective"** (Koruca et al., 2023) discusses methodologies that incorporate staff preferences and flexible scheduling, with results applicable to care workforce settings[^5_6].
- The **World Health Organisation's "Health and care workforce planning tools"** (2025) provides a rapid review of available planning and scheduling tools, mapping their strengths and limitations for health and care workforce management[^5_7].

These reviews collectively cover mathematical models, algorithmic approaches, and practical considerations for scheduling in care environments.

[^5_1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8623715/

[^5_2]: https://www.sciencedirect.com/science/article/pii/S002074892200150X

[^5_3]: https://www.sciencedirect.com/science/article/abs/pii/S2211692314200300

[^5_4]: https://www.mdpi.com/2076-3417/15/7/3610

[^5_5]: https://pubmed.ncbi.nlm.nih.gov/39213408/

[^5_6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9972317/

[^5_7]: https://www.who.int/publications/i/item/9789240106277

[^5_8]: https://journal.ilpnetwork.org/articles/10.31389/jltc.66


---

# Would it be possible to create an MVP riding classes to implement the elements of this problem? What parameters need to be provided or estimated for such a model?

Yes, it is possible to create a Minimum Viable Product (MVP) using classes to implement the core elements of the care workforce scheduling problem. This MVP would focus on the essential scheduling logic and data structures, allowing for rapid prototyping and iteration[^6_1][^6_2][^6_3].

## Key Parameters to Provide or Estimate

To build a functional MVP, you need to define or estimate the following parameters:

- **Carers**
    - Unique ID or name
    - Skills/qualifications
    - Availability (time slots)
    - Maximum working hours
    - Preferences (optional, e.g., preferred clients)
- **Care Recipients**
    - Unique ID or name
    - Care requirements (skills needed)
    - Required time slots for care
    - Preferences for carers (optional)
- **Time Slots**
    - Definition of scheduling periods (e.g., hourly, daily)
    - Total number of periods to schedule
- **Assignments**
    - Cost or penalty for each possible carer-recipient-time assignment (can include travel, overtime, or preference penalties)
- **Constraints**
    - Skill matching (carer must have required skills)
    - Carer availability and maximum hours
    - Recipient care requirements per slot
    - Optional: recipient/carer preferences, continuity of care, travel limits


## MVP Class Structure (Example, in Python-like pseudocode)

```python
class Carer:
    def __init__(self, id, skills, availability, max_hours, preferences=None):
        ...

class Recipient:
    def __init__(self, id, required_skills, required_slots, preferences=None):
        ...

class TimeSlot:
    def __init__(self, slot_id, datetime):
        ...

class Assignment:
    def __init__(self, carer, recipient, time_slot, cost):
        ...

class Scheduler:
    def __init__(self, carers, recipients, time_slots, assignments, constraints):
        ...
    def optimise_schedule(self):
        ...
```


## MVP Development Steps

1. **Identify the minimum set of features**: Focus on core scheduling and constraint logic[^6_1][^6_3].
2. **Collect or estimate input data**: Gather sample data for carers, recipients, and time slots.
3. **Implement scheduling logic**: Start with a simple greedy or rule-based assignment, then iterate.
4. **Test and refine**: Use real or synthetic data to validate the model and gather feedback[^6_2][^6_3].

This approach allows you to quickly build, test, and improve a basic scheduling system before adding more complex features like advanced optimisation, UI, or integrations[^6_1][^6_2][^6_3][^6_5].



[^6_1]: https://www.myshyft.com/blog/minimum-viable-schedule/

[^6_2]: https://omnivatelehealth.com/blog/launch-minimum-viable-product-mvp/

[^6_3]: https://www.spaceotechnologies.com/blog/minimum-viable-product-for-healthcare/

[^6_4]: https://www.cleverdevsoftware.com/blog/minimum-viable-product-healthcare

[^6_5]: https://www.griddynamics.com/blog/building-workforce-schedule-optimisation-solution

[^6_6]: https://www.opm.gov/policy-data-oversight/human-capital-framework/reference-materials/talent-management/workforce-planning-guide.pdf

[^6_7]: https://tateeda.com/blog/how-to-build-an-mvp-for-a-healthcare-product

[^6_8]: https://www.salesforce.com/blog/minimum-viable-to-minimum-valuable-product/


---

# Can you provide indicative sample data for each class

Indicative sample data for each class in a care workforce scheduling MVP, based on typical real-world home care or disability service scenarios[^7_5][^7_1][^7_2].

## Carer

```python
carers = [
    {
        "id": "C001",
        "name": "Alex Smith",
        "skills": ["elderly_care", "medication_management"],
        "availability": ["2025-06-26 09:00", "2025-06-26 13:00", "2025-06-26 15:00"],
        "max_hours": 8,
        "preferences": ["P001", "P003"]
    },
    {
        "id": "C002",
        "name": "Jamie Lee",
        "skills": ["disability_support", "personal_care"],
        "availability": ["2025-06-26 10:00", "2025-06-26 14:00", "2025-06-26 16:00"],
        "max_hours": 6,
        "preferences": ["P002"]
    }
]
```


## Recipient

```python
recipients = [
    {
        "id": "P001",
        "name": "Mary Johnson",
        "required_skills": ["elderly_care"],
        "required_slots": ["2025-06-26 09:00", "2025-06-26 13:00"],
        "preferences": ["C001"]
    },
    {
        "id": "P002",
        "name": "Sam Wilson",
        "required_skills": ["disability_support"],
        "required_slots": ["2025-06-26 10:00", "2025-06-26 14:00"],
        "preferences": ["C002"]
    }
]
```


## Time Slot

```python
time_slots = [
    {"slot_id": "T001", "datetime": "2025-06-26 09:00"},
    {"slot_id": "T002", "datetime": "2025-06-26 10:00"},
    {"slot_id": "T003", "datetime": "2025-06-26 13:00"},
    {"slot_id": "T004", "datetime": "2025-06-26 14:00"}
]
```


## Assignment

```python
assignments = [
    {
        "carer_id": "C001",
        "recipient_id": "P001",
        "time_slot_id": "T001",
        "cost": 10
    },
    {
        "carer_id": "C002",
        "recipient_id": "P002",
        "time_slot_id": "T002",
        "cost": 12
    }
]
```

These samples provide a clear structure for implementing a basic workforce scheduling system for care services.

[^7_1]: https://www.ibm.com/docs/en/icos/22.1.0?topic=models-presenting-nurse-scheduling-example

[^7_2]: https://www.kimedics.com/blog/healthcare-staff-scheduling

[^7_3]: https://www.onshift.com/the-complete-guide-to-workforce-scheduling-for-senior-care-organisations

[^7_4]: https://www.ahrq.gov/sites/default/files/wysiwyg/professionals/systems/primary-care/workforce-financing/case_example_6.pdf

[^7_5]: https://pure.strath.ac.uk/ws/portalfiles/portal/92664633/Readme.pdf

[^7_6]: https://unrubble.com/blog/healthcare-staff-scheduling

[^7_7]: https://www.aonl.org/sites/default/files/aone/workforce-planning-model.pdf

[^7_8]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9065499/


---

# So it would seem that a carer and a recipient should ideally provide their schedule (probably in like a Calendar type object?) for up to 2 weeks in advance (although maybe also with a separate holiday schedule?)

Yes, that approach is both practical and widely used in workforce scheduling systems for care.

**Summary of Best Practices:**

- **Calendar Objects for Schedules:**
Carers and recipients (clients) should provide their availability and care needs as time slots within a calendar-like structure. This allows the system to easily visualise and manage overlapping or conflicting times, and supports recurring appointments or exceptions[^8_2][^8_3][^8_5].
- **Planning Horison:**
Providing schedules for up to 2 weeks in advance is common and helps with early planning, but the system should also allow for last-minute changes or ad-hoc requests[^8_4][^8_5].
- **Holiday/Leave Schedules:**
Separate holiday or leave schedules (e.g., public holidays, planned absences) should be integrated so that the system automatically blocks unavailable times and avoids scheduling conflicts[^8_2][^8_3].
- **Flexibility and Transparency:**
Allowing carers and recipients to update their availability and preferences, and making the scheduling process transparent, improves satisfaction and reduces conflicts[^8_2][^8_5].

**Example Data Structure:**

```python
carer_calendar = {
    "id": "C001",
    "name": "Alex Smith",
    "availability": [
        {"start": "2025-06-26 09:00", "end": "2025-06-26 12:00"},
        {"start": "2025-06-26 13:00", "end": "2025-06-26 16:00"}
    ],
    "unavailable": [
        {"start": "2025-06-27", "end": "2025-06-29", "reason": "holiday"}
    ]
}

recipient_calendar = {
    "id": "P001",
    "name": "Mary Johnson",
    "required_slots": [
        {"start": "2025-06-26 09:00", "end": "2025-06-26 10:30"},
        {"start": "2025-06-26 13:00", "end": "2025-06-26 14:30"}
    ]
}
```

This structure supports both routine and exceptional scheduling needs.



[^8_1]: https://www.onshift.com/the-complete-guide-to-workforce-scheduling-for-senior-care-organisations

[^8_2]: https://unrubble.com/blog/healthcare-staff-scheduling

[^8_3]: https://www.supersaas.com/info/workforce-planning-system

[^8_4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8623715/

[^8_5]: https://tcpsoftware.com/articles/employee-scheduling-in-healthcare/

[^8_6]: https://www.onshift.com/products/workforce-management-software/workforce-scheduling

[^8_7]: https://www.inovalon.com/resource/how-innovative-workforce-systems-transform-scheduling-processes-and-improve-staff-morale/

[^8_8]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9972317/


---

# UI for a carer app and also client app

For both carer and client (recipient) apps, the UI should focus on making scheduling, availability, and communication clear and intuitive. Here’s what each might involve:

## Carer App UI

- **Availability Calendar:** Carers can easily select, edit, and submit their available time slots for up to two weeks (with options for recurring patterns and holiday/leave blocks)[^9_5][^9_7].
- **Shift Assignments:** View upcoming assigned shifts/visits, including client details, location, and care requirements[^9_3][^9_5].
- **Notifications:** Alerts for new assignments, changes, cancellations, or messages from clients/admins[^9_6][^9_7].
- **Profile \& Preferences:** Manage skills, preferred clients, and update personal info.
- **Communication:** In-app chat or messaging with coordinators or clients for clarifications[^9_5][^9_7].
- **Leave/Holiday Management:** Submit and view approved leave periods or unavailable dates.


## Client (Recipient) App UI

- **Care Request Calendar:** Clients specify when they need care, see confirmed bookings, and request or adjust time slots[^9_3][^9_6].
- **Carer Preferences:** Indicate preferred carers and view which carer is assigned for each visit.
- **Notifications:** Reminders for upcoming visits, carer changes, or messages[^9_6].
- **Communication:** Secure messaging with carers or care coordinators for special requests or updates[^9_5][^9_7].
- **Profile:** Manage care needs, preferences, and emergency contacts.


## Shared UI Features

- **Clear Calendar Views:** Both apps should offer easy-to-navigate calendar or timeline views for all scheduled and available slots[^9_3][^9_5].
- **Easy Rescheduling:** Simple tools to request or accept changes to bookings[^9_3][^9_6].
- **Accessibility:** Large buttons, readable fonts, and support for users with disabilities[^9_8].

> “Effective appointment scheduling UI involves easy-to-navigate calendars, efficient display of time slots, and easy appointment confirmation and management… flexibility to reschedule or cancel appointments easily is crucial.”[^9_3]

You can find visual inspiration for these interfaces on platforms like Dribbble and Figma, which showcase staff and patient scheduling app designs[^9_1][^9_2][^9_5].



[^9_1]: https://dribbble.com/search/patient-scheduling

[^9_2]: https://dribbble.com/search/staff-scheduling-software

[^9_3]: https://www.koruux.com/50-examples-of-healthcare-UI/

[^9_4]: https://www.imaginarycloud.com/blog/3-healthcare-apps-with-the-best-ui-ux-design

[^9_5]: https://www.figma.com/community/file/1426199074507218546/employee-availability-app-admin-employee-mobile-app-ui

[^9_6]: https://www.matellio.com/blog/patient-scheduling-app-development/

[^9_7]: https://www.linkedin.com/pulse/designing-effective-scheduling-interfaces-healthcare-koshy-saji-2v1sc

[^9_8]: https://fuselabcreative.com/healthcare-app-ui-ux-design-best-practices/


---

# Can you build sample UI apps in Streamlit to demo the look and feel of these features

Yes, you can quickly build demo UI apps in Streamlit to showcase the look and feel of both carer and client (recipient) scheduling features. Streamlit is ideal for rapid prototyping with interactive components like forms, calendars, and notifications[^10_1][^10_2][^10_5].

## Key UI Features to Demo in Streamlit

- **Interactive Calendar:** Use the `streamlit_calendar` component to display and select availability or care needs in a familiar calendar view[^10_3].
- **Availability Forms:** Carers and clients can input or update their available/unavailable times using forms and date/time pickers[^10_2].
- **Shift/Visit Overview:** Display upcoming assignments or bookings in tables or calendar views.
- **Notifications/Reminders:** Use Streamlit’s message and alert components for updates.
- **Preference Management:** Dropdowns or multiselects for preferred carers/clients.


## Example: Carer Availability UI (Streamlit)

```python
import streamlit as st
from datetime import datetime, timedelta

st.title("Carer Availability")

# Calendar selection (using streamlit_calendar or st.date_input for demo)
st.subheader("Select your available time slots")
available_dates = st.date_input("Available dates", [])

# Time slot selection
start_time = st.time_input("Start time", value=datetime.now().time())
end_time = st.time_input("End time", value=(datetime.now() + timedelta(hours=1)).time())

if st.button("Add Availability"):
    st.success(f"Added availability: {available_dates} from {start_time} to {end_time}")

# Display current availabilities (dummy data)
st.table([
    {"Date": "2025-06-26", "Start": "09:00", "End": "12:00"},
    {"Date": "2025-06-27", "Start": "13:00", "End": "16:00"}
])
```


## Example: Client Request UI (Streamlit)

```python
st.title("Request Care")

# Care needs form
care_date = st.date_input("Date needed")
care_start = st.time_input("Start time")
care_end = st.time_input("End time")
preferred_carer = st.selectbox("Preferred Carer", ["Alex Smith", "Jamie Lee", "No preference"])

if st.button("Request Care"):
    st.success(f"Requested care on {care_date} from {care_start} to {care_end} (Preferred: {preferred_carer})")
```


## Calendar Integration

For richer calendar views, use the `streamlit_calendar` package[^10_3]:

```python
# pip install streamlit-calendar
from streamlit_calendar import calendar

calendar(events=[...], options={...})
```


## Summary

- Streamlit enables rapid prototyping of scheduling UIs with calendars, forms, and tables[^10_1][^10_2][^10_3][^10_5].
- You can build both carer and client demo apps in a few lines of code.
- For production, you’d refine data handling, authentication, and user experience.


---


# Comprehensive Guide to Care Workforce Scheduling Systems

## 1. Introduction

Workforce scheduling in care services—such as in-home aged care and disability support—is a complex, constraint-rich problem. The goal is to match carers to clients (recipients) in a way that satisfies regulatory, operational, and personal requirements, while also optimising for cost, continuity, and satisfaction.


## 5. User Interface (UI) Considerations

### 5.1. Carer App

- **Availability Calendar:** Mark available/unavailable slots, recurring patterns, holidays.
- **Shift Assignments:** View upcoming visits and details.
- **Notifications:** For new assignments or changes.
- **Profile \& Preferences:** Update skills, preferences, and personal info.
- **Communication:** Messaging with coordinators or clients.


### 5.2. Client App

- **Care Request Calendar:** Specify care needs, see bookings, request changes.
- **Carer Preferences:** Indicate and view preferred carers.
- **Notifications:** Reminders and updates.
- **Communication:** Messaging with carers or coordinators.
- **Profile:** Manage care needs and contacts.


### 5.3. Shared Features

- **Clear calendar/timeline views**
- **Easy rescheduling**
- **Accessibility options**


## 6. Streamlit Demo UI Concepts

### Carer Availability Example

```python
import streamlit as st
from datetime import datetime, timedelta

st.title("Carer Availability")

st.subheader("Select your available time slots")
available_dates = st.date_input("Available dates", [])

start_time = st.time_input("Start time", value=datetime.now().time())
end_time = st.time_input("End time", value=(datetime.now() + timedelta(hours=1)).time())

if st.button("Add Availability"):
    st.success(f"Added availability: {available_dates} from {start_time} to {end_time}")

st.table([
    {"Date": "2025-06-26", "Start": "09:00", "End": "12:00"},
    {"Date": "2025-06-27", "Start": "13:00", "End": "16:00"}
])
```


### Client Care Request Example

```python
st.title("Request Care")

care_date = st.date_input("Date needed")
care_start = st.time_input("Start time")
care_end = st.time_input("End time")
preferred_carer = st.selectbox("Preferred Carer", ["Alex Smith", "Jamie Lee", "No preference"])

if st.button("Request Care"):
    st.success(f"Requested care on {care_date} from {care_start} to {care_end} (Preferred: {preferred_carer})")
```


## 7. Further Reading and Research

- **Lieder et al. (2015):** Task scheduling in long-term care facilities: A client-centered approach.
- **Morse et al. (2024):** Centralised Scheduling of Nursing Staff: A Rapid Review.
- **Koruca et al. (2023):** Personalised staff-scheduling method with work-life balance.
- **WHO (2025):** Health and care workforce planning tools.


## 8. Next Steps

- **Prototype the MVP** using the above data structures and UI concepts.
- **Iterate** based on user feedback from carers, clients, and coordinators.
- **Expand** with advanced scheduling logic, notifications, and integrations as needed.

**This document provides a blueprint for building, prototyping, and understanding a care workforce scheduling system, from mathematical formulation to UI and demo implementation.** If you need more detailed code, a specific feature, or further reading, just ask!

---

You are correct: in real-world care scheduling, a client's care need at a given time is not just a binary indicator, but is often **qualified by the type(s) of care required** (e.g., nursing, mobility, medication assistance). Similarly, a provider's ability to meet a need depends on their skills and availability for specific care types.

## Generalising the Indicator Function for Care Types

To capture this, the indicator function should be indexed not just by client and time, but also by care type or skill. This makes the model flexible and expressive.

### Notation

Let:

- $\mathcal{C}$: Set of clients
- $\mathcal{P}$: Set of care providers
- $\mathcal{T}$: Set of time slots
- $\mathcal{S}$: Set of care types/skills

Define indicator functions:

- **$\mathcal{I}_{\text{need}}(c, s, t)$**: 1 if client $c$ needs care type/skill $s$ at time $t$, 0 otherwise
- **$\mathcal{I}_{\text{avail}}(p, t)$**: 1 if provider $p$ is available at time $t$, 0 otherwise
- **$\mathcal{I}_{\text{skill}}(p, s)$**: 1 if provider $p$ can deliver care type/skill $s$, 0 otherwise


### Constraints (Illustrative)

- **Client Care Need Coverage:**

$$
\sum_{p \in \mathcal{P}} \mathcal{I}_{\text{skill}}(p, s) \cdot x_{p,c,s,t} \geq \mathcal{I}_{\text{need}}(c, s, t) \quad \forall c, s, t
$$

Here, $x_{p,c,s,t}$ is 1 if provider $p$ is assigned to client $c$ for skill $s$ at time $t$.
- **Provider Assignment Only If Available and Skilled:**

$$
x_{p,c,s,t} \leq \mathcal{I}_{\text{avail}}(p, t) \cdot \mathcal{I}_{\text{skill}}(p, s)
$$


### Summary Table

| Symbol | Description | Example |
| :-- | :-- | :-- |
| $\mathcal{I}_{\text{need}}(c,s,t)$ | Client $c$ needs care type $s$ at time $t$ | $\mathcal{I}_{\text{need}}(\text{Mary}, \text{medication}, 09:00) = 1$ |
| $\mathcal{I}_{\text{avail}}(p,t)$ | Provider $p$ is available at time $t$ | $\mathcal{I}_{\text{avail}}(\text{Bob}, 09:00) = 1$ |
| $\mathcal{I}_{\text{skill}}(p,s)$ | Provider $p$ can deliver care type $s$ | $\mathcal{I}_{\text{skill}}(\text{Bob}, \text{medication}) = 1$ |

### Conclusion

This generalisation ensures your model can handle multiple care types/skills per client and per provider, and that assignments are only made when both need and capability align[^16_1][^16_2]. This approach is rigorous, compact, and extensible for real-world care scheduling.



[^16_1]: https://en.wikipedia.org/wiki/Indicator_function

[^16_2]: https://math.stackexchange.com/questions/1294113/proper-use-of-indicator-function

[^16_3]: https://help.sap.com/docs/SAP_ERP/486822dda1ae45dca999ef5104bd35c1/12a8ce5314894208e10000000a174cb4.html?version=6.03.latest

[^16_4]: https://www.mdpi.com/2227-7390/11/3/750

[^16_5]: https://www.cms.gov/medicare/provider-enrollment-and-certification/qapi/downloads/measindicatdevwksdebedits.pdf

[^16_6]: https://ecampusontario.pressbooks.pub/math3080prep/chapter/1-3-using-function-notation/

[^16_7]: https://help.sap.com/docs/SAP_S4HANA_ON-PREMISE/c9b5e9de6e674fb99fff88d72c352291/ee6ead83d6084103bcddb5fcdd64ce0d.html

[^16_8]: https://www.statlect.com/fundamentals-of-probability/indicator-functions


---

# here does t = 9:00 (mean a 9 to 10 time slot?)

Yes, in most healthcare and care workforce scheduling systems, a time slot such as “9:00” typically represents a block of time—most commonly an interval like 9:00 to 10:00, but the exact duration depends on how the slots are defined in your system[^17_2][^17_4][^17_5][^17_8].

**Key points:**

- **Time Slot Definition:**
A time slot is a discrete, bookable interval. In many systems, slots are 15, 30, or 60 minutes long[^17_2][^17_5][^17_8]. If your schedule defines slots as “9:00,” it usually means the interval from 9:00 to 10:00 (if using hourly slots), or from 9:00 to 9:30 (if using half-hour slots), etc.
- **Explicit Duration:**
It’s best practice to define both the start time and the duration of each slot, e.g., “9:00–10:00” or with a slot length parameter[^17_8][^17_10].
- **Modeling:**
In your mathematical model, $t$ should represent the start of a time interval, and you should specify the slot length (e.g., 1 hour). So “$t = 9:00$” means the slot from 9:00 to 10:00 if you’re using hourly slots.

**Example:**

- If $\mathcal{T} = \{\text{09:00}, \text{10:00}, \text{11:00}\}$ and slots are 1 hour, then:
    - $t = 9:00$ means 9:00–10:00
    - $t = 10:00$ means 10:00–11:00

**Summary Table Example:**


| t (slot) | Slot Start | Slot End | Duration |
| :-- | :-- | :-- | :-- |
| 1 | 09:00 | 10:00 | 1 hour |
| 2 | 10:00 | 11:00 | 1 hour |

**References:**

- Time-slot scheduling is the most common approach, and each slot is a defined interval, not a single instant[^17_2][^17_4][^17_5][^17_8].

If your application needs variable slot lengths (e.g., 15, 30, or 60 minutes), you can generalise the model by defining both start and end times for each slot, or by including a slot duration parameter[^17_8][^17_10].



[^17_1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6452836/

[^17_2]: https://www.verint.com/blog/what-are-the-8-different-types-of-appointment-scheduling/

[^17_3]: https://carevision.com/scheduling-care-guide/

[^17_4]: https://www.sciencedirect.com/science/article/abs/pii/S0360835219301202

[^17_5]: https://www.ohsu.edu/sites/default/files/2021-10/10.28 0830_0930%20Exploring%20New%20RHC%20Scheduling%20Models%20to%20Improve%20Patient%20Access.pdf

[^17_6]: https://hl7.org/fhir/R4/schedule.html

[^17_7]: https://edhub.ama-assn.org/steps-forward/module/2810481

[^17_8]: https://www.acponline.org/sites/default/files/documents/running_practice/patient_care/pat_sched.pdf

[^17_9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5228137/

[^17_10]: https://pubmed.ncbi.nlm.nih.gov/32952603/


---

# Lets go with start time and slot length (assume fixed eg 30 mins) and there can be multiple slots for an "appointment"

Here’s how you can structure your model and notation to accommodate:

- **Time slots defined by a start time and fixed slot length (e.g., 30 minutes)**
- **Appointments that may span multiple consecutive slots**


## Time Slot Structure

Let each time slot $t$ be defined by:

- $\text{start}(t)$: the start time of slot $t$ (e.g., 09:00)
- $\ell$: fixed slot length (e.g., 30 minutes)

So, slot $t$ covers the interval:

$$
[\text{start}(t),\ \text{start}(t) + \ell)
$$

## Appointments Spanning Multiple Slots

If a client requires an appointment of duration $d$ (e.g., 90 minutes), this will span $k = d / \ell$ consecutive slots.

## Model Adjustments

### Sets

- $\mathcal{T}$: Ordered set of time slots, each with $\text{start}(t)$


### Parameters

- $\ell$: Slot length (e.g., 30 minutes)
- $d_{c,s,a}$: Duration required for appointment $a$ (of care type $s$ for client $c$)
- $\mathcal{A}_c$: Set of appointments for client $c$


### Indicator Functions

- $\mathcal{I}_{\text{need}}(c, s, t)$: 1 if client $c$ needs care type $s$ at time slot $t$ (as part of an appointment), 0 otherwise
- $\mathcal{I}_{\text{avail}}(p, t)$: 1 if provider $p$ is available at time slot $t$
- $\mathcal{I}_{\text{skill}}(p, s)$: 1 if provider $p$ can deliver care type $s$


### Decision Variable

- $x_{p, c, s, t}$: 1 if provider $p$ is assigned to client $c$ for care type $s$ at slot $t$


### Constraints

- **Appointment Coverage:**
For each appointment $a$ for client $c$, care type $s$, and required consecutive slots $\{t_1, t_2, ..., t_k\}$:

$$
\sum_{p \in \mathcal{P}} \mathcal{I}_{\text{skill}}(p, s) \cdot x_{p, c, s, t_j} \geq \mathcal{I}_{\text{need}}(c, s, t_j)
$$

for all $t_j$ in the appointment.
- **Provider Assignment:**

$$
x_{p, c, s, t} \leq \mathcal{I}_{\text{avail}}(p, t) \cdot \mathcal{I}_{\text{skill}}(p, s)
$$
- **Provider Cannot Double Book:**

$$
\sum_{c \in \mathcal{C}} \sum_{s \in \mathcal{S}} x_{p, c, s, t} \leq 1 \quad \forall p, t
$$
- **Provider Max Hours:**

$$
\sum_{c \in \mathcal{C}} \sum_{s \in \mathcal{S}} \sum_{t \in \mathcal{T}} x_{p, c, s, t} \cdot \ell \leq H_p
$$


## Example

- Slot length $\ell = 30$ min
- Slots: 09:00, 09:30, 10:00, 10:30, etc.
- Appointment: Client Mary needs 90 minutes of medication care starting at 09:00
- This requires $k = 3$ consecutive slots: 09:00, 09:30, 10:00

So, $\mathcal{I}_{\text{need}}(\text{Mary}, \text{medication}, t) = 1$ for $t \in \{\text{09:00}, \text{09:30}, \text{10:00}\}$

## Summary Table (Example)

| Slot $t$ | Start | End | $\ell$ (min) |
| :-- | :-- | :-- | :-- |
| 1 | 09:00 | 09:30 | 30 |
| 2 | 09:30 | 10:00 | 30 |
| 3 | 10:00 | 10:30 | 30 |

This approach supports variable appointment lengths (in multiples of the slot length), ensures clarity in scheduling, and is robust for real-world care scenarios.

---

# Comprehensive Guide to Care Workforce Scheduling Systems

## Executive Summary

This document provides a rigorous, modern formulation for care workforce scheduling—covering mathematical modeling, data structures, user interface considerations, and an overview of potential vendor solutions (both open source and commercial). The focus is on home care and disability services, with an emphasis on flexibility, clarity, and extensibility for real-world requirements[^19_1].

## 1. Problem Overview

Workforce scheduling in care services involves assigning care providers to clients across time slots, subject to constraints such as provider skills, client care needs, availability, preferences, and regulatory limits. The model must support:

- Multiple care types/skills per client and provider
- Appointments spanning multiple time slots
- Preference and cost optimisation


## 2. Mathematical Formulation

### 2.1 Sets and Indices

- **$\mathcal{P}$**: Set of Care Providers (indexed by $p$)
- **$\mathcal{C}$**: Set of Clients (indexed by $c$)
- **$\mathcal{T}$**: Set of Time Slots (indexed by $t$), each with a start time $\text{start}(t)$
- **$\mathcal{S}$**: Set of Skills/Care Types (indexed by $s$)
- **$\ell$**: Fixed slot length (e.g., 30 minutes)


### 2.2 Indicator Functions

- **$\mathcal{I}_{\text{avail}}(p, t)$**: 1 if provider $p$ is available at time slot $t$, 0 otherwise
- **$\mathcal{I}_{\text{need}}(c, s, t)$**: 1 if client $c$ needs care type $s$ at time slot $t$, 0 otherwise
- **$\mathcal{I}_{\text{skill}}(p, s)$**: 1 if provider $p$ can deliver care type $s$, 0 otherwise


### 2.3 Parameters

- **$H_p$**: Maximum working hours for provider $p$ in the planning period (e.g., per day)
- **$C_{p,c,s,t}$**: Cost (e.g., dollars per slot) for assigning provider $p$ to client $c$ for care type $s$ at time slot $t$
- **$\pi_{c,p}$**: Preference score of client $c$ for provider $p$ (higher is better)


### 2.4 Decision Variable

- **$x_{p,c,s,t} \in \{0,1\}$**: 1 if provider $p$ is assigned to client $c$ for care type $s$ at slot $t$, 0 otherwise


### 2.5 Objective Function

$$
\min \sum_{p \in \mathcal{P}} \sum_{c \in \mathcal{C}} \sum_{s \in \mathcal{S}} \sum_{t \in \mathcal{T}} \left( C_{p,c,s,t} \cdot x_{p,c,s,t} - \lambda \, \pi_{c,p} \cdot x_{p,c,s,t} \right)
$$

where $\lambda$ balances cost and preference.

### 2.6 Constraints

1. **Provider Availability and Skill:**

$$
x_{p,c,s,t} \leq \mathcal{I}_{\text{avail}}(p, t) \cdot \mathcal{I}_{\text{skill}}(p, s)
$$
2. **Client Care Need Coverage:**

$$
\sum_{p \in \mathcal{P}} x_{p,c,s,t} \geq \mathcal{I}_{\text{need}}(c, s, t) \quad \forall c, s, t
$$
3. **Provider Cannot Double Book:**

$$
\sum_{c \in \mathcal{C}} \sum_{s \in \mathcal{S}} x_{p,c,s,t} \leq 1 \quad \forall p, t
$$
4. **Provider Maximum Working Hours:**

$$
\sum_{c \in \mathcal{C}} \sum_{s \in \mathcal{S}} \sum_{t \in \mathcal{T}} x_{p,c,s,t} \cdot \ell \leq H_p \quad \forall p
$$

## 3. Data Structure Examples

### 3.1. Time Slot Example

| Slot ID | Start | End | Length (min) |
| :-- | :-- | :-- | :-- |
| 1 | 09:00 | 09:30 | 30 |
| 2 | 09:30 | 10:00 | 30 |
| ... | ... | ... | ... |

### 3.2. Appointment Example

- Client Mary needs 90 minutes of medication care starting at 09:00:
    - $\mathcal{I}_{\text{need}}(\text{Mary}, \text{medication}, t) = 1$ for $t \in \{\text{09:00}, \text{09:30}, \text{10:00}\}$


### 3.3. Assignment Variable

- $x_{\text{Alice},\ \text{Mary},\ \text{medication},\ 09:00} = 1$ means Alice is assigned to Mary for medication at 09:00–09:30.


## 4. User Interface Considerations

### 4.1. Care Provider (Carer) App

- **Availability Calendar:** Mark available/unavailable slots, holidays, and preferences.
- **Shift Assignments:** View upcoming visits with client/care details.
- **Notifications:** For new assignments or changes.
- **Profile Management:** Skills, preferences, personal info.
- **Leave Management:** Submit/view leave periods.


### 4.2. Client App

- **Care Request Calendar:** Specify care needs, durations, and preferred providers.
- **Booking Overview:** See confirmed and pending bookings.
- **Notifications:** Reminders and updates.
- **Preference Setting:** Indicate preferred providers.


### 4.3. Shared Features

- **Calendar View:** Navigate all scheduled and available slots.
- **Easy Rescheduling:** Request or accept changes.
- **Accessibility:** Large buttons, readable fonts, support for disabilities.


## 5. Implementation Notes

- **Time slots are defined by start time and fixed length (e.g., 30 min).**
- **Appointments can span multiple consecutive slots.**
- **Indicator functions generalise binary needs, skills, and availability, indexed by all relevant dimensions.**
- **All constraints and objective terms can be extended for overtime, travel, continuity, etc.**


## 6. Notation Table with Examples

| Symbol | Description | Example |
| :-- | :-- | :-- |
| $\mathcal{P}$ | Care Providers | Alice, Bob |
| $\mathcal{C}$ | Clients | Mary, Sam |
| $\mathcal{T}$ | Time slots (start times) | 09:00, 09:30, 10:00 |
| $\mathcal{S}$ | Skills/Care Types | medication, mobility |
| $\mathcal{I}_{\text{avail}}(p, t)$ | Provider available at time $t$ | $\mathcal{I}_{\text{avail}}(\text{Alice}, 09:00) = 1$ |
| $\mathcal{I}_{\text{need}}(c, s, t)$ | Client needs care type at time $t$ | $\mathcal{I}_{\text{need}}(\text{Mary}, \text{medication}, 09:00) = 1$ |
| $\mathcal{I}_{\text{skill}}(p, s)$ | Provider can deliver care type $s$ | $\mathcal{I}_{\text{skill}}(\text{Bob}, \text{medication}) = 1$ |
| $H_p$ | Max hours per planning period | $H_{\text{Alice}} = 8$ (hours/day) |
| $C_{p,c,s,t}$ | Cost per slot | $C_{\text{Alice}, \text{Mary}, \text{medication}, 09:00} = 30$ (dollars/slot) |
| $\pi_{c,p}$ | Client’s preference for provider | $\pi_{\text{Mary}, \text{Alice}} = 10$ |
| $x_{p,c,s,t}$ | Assignment variable | $x_{\text{Alice}, \text{Mary}, \text{medication}, 09:00} = 1$ |

## Appendix: Potential Vendor Solutions

### Open Source Solutions

| Name/Platform | Description | Notes |
| :-- | :-- | :-- |
| **OptaPlanner** | Advanced open-source constraint solver for employee rostering and scheduling | Java-based, highly customisable |
| **Staffjoy** | Open-source scheduling platform (archived, but code available) | Python/Go, web-based |
| **Auto Shift Planner** | Open-source employee shift planning tool | Python, heuristic-based |
| **Cal.com** | Open-source appointment scheduling, can be adapted for workforce needs | Node.js, calendar focus |

### Commercial Solutions

| Name/Platform | Description | Notes |
| :-- | :-- | :-- |
| **Skedulo** | Cloud-based mobile workforce management, strong in healthcare/home care | Commercial, API-rich |
| **Allocate Software** | Workforce management for healthcare, hospitals, and care providers | Enterprise-level, UK/AU focus |
| **AlayaCare** | End-to-end home care management, including scheduling and compliance | Cloud-based, care sector specialist |
| **Deputy** | Popular workforce scheduling, used in healthcare and hospitality | Commercial, flexible integrations |
| **Rotageek** | AI-driven scheduling, used by NHS and care providers | Commercial, strong optimisation |

**Note:** Open source tools often require more customisation and technical resources, while commercial solutions offer robust support, integrations, and compliance features out of the box.

## References \& Further Reading

- Lieder et al. (2015): Task scheduling in long-term care facilities: A client-centered approach.
- Morse et al. (2024): Centralised Scheduling of Nursing Staff: A Rapid Review.
- Koruca et al. (2023): Personalised staff-scheduling method with work-life balance.
- WHO (2025): Health and care workforce planning tools.


---

# Comprehensive Guide to Care Workforce Scheduling Systems

## Executive Summary

This document presents a modern, extensible approach to care workforce scheduling, covering mathematical modeling, data structures, user interface (UI) considerations, current industry trends, and an overview of both open source and commercial vendor solutions. The focus is on home care and disability services, with an emphasis on practical implementation and optimisation.

## 1. Problem Overview

Care workforce scheduling involves assigning care providers to clients across time slots, subject to constraints such as provider skills, client care needs, availability, preferences, and regulatory limits. The model must support:

- Multiple care types/skills per client and provider
- Appointments spanning multiple time slots
- Preference and cost optimisation
- Compliance with labor and industry regulations


## 2. Mathematical Formulation

### 2.1 Sets and Indices

- **$\mathcal{P}$**: Set of Care Providers (indexed by $p$)
- **$\mathcal{C}$**: Set of Clients (indexed by $c$)
- **$\mathcal{T}$**: Set of Time Slots (indexed by $t$), each with a start time $\text{start}(t)$
- **$\mathcal{S}$**: Set of Skills/Care Types (indexed by $s$)
- **$\ell$**: Fixed slot length (e.g., 30 minutes)


### 2.2 Indicator Functions

- **$\mathcal{I}_{\text{avail}}(p, t)$**: 1 if provider $p$ is available at time slot $t$, 0 otherwise
- **$\mathcal{I}_{\text{need}}(c, s, t)$**: 1 if client $c$ needs care type $s$ at time slot $t$, 0 otherwise
- **$\mathcal{I}_{\text{skill}}(p, s)$**: 1 if provider $p$ can deliver care type $s$, 0 otherwise


### 2.3 Parameters

- **$H_p$**: Maximum working hours for provider $p$ in the planning period (e.g., per day)
- **$C_{p,c,s,t}$**: Cost (e.g., dollars per slot) for assigning provider $p$ to client $c$ for care type $s$ at time slot $t$
- **$\pi_{c,p}$**: Preference score of client $c$ for provider $p$ (higher is better)


### 2.4 Decision Variable

- **$x_{p,c,s,t} \in \{0,1\}$**: 1 if provider $p$ is assigned to client $c$ for care type $s$ at slot $t$, 0 otherwise


### 2.5 Objective Function

$$
\min \sum_{p \in \mathcal{P}} \sum_{c \in \mathcal{C}} \sum_{s \in \mathcal{S}} \sum_{t \in \mathcal{T}} \left( C_{p,c,s,t} \cdot x_{p,c,s,t} - \lambda \, \pi_{c,p} \cdot x_{p,c,s,t} \right)
$$

where $\lambda$ balances cost and preference.

### 2.6 Constraints

1. **Provider Availability and Skill:**

$$
x_{p,c,s,t} \leq \mathcal{I}_{\text{avail}}(p, t) \cdot \mathcal{I}_{\text{skill}}(p, s)
$$
2. **Client Care Need Coverage:**

$$
\sum_{p \in \mathcal{P}} x_{p,c,s,t} \geq \mathcal{I}_{\text{need}}(c, s, t) \quad \forall c, s, t
$$
3. **Provider Cannot Double Book:**

$$
\sum_{c \in \mathcal{C}} \sum_{s \in \mathcal{S}} x_{p,c,s,t} \leq 1 \quad \forall p, t
$$
4. **Provider Maximum Working Hours:**

$$
\sum_{c \in \mathcal{C}} \sum_{s \in \mathcal{S}} \sum_{t \in \mathcal{T}} x_{p,c,s,t} \cdot \ell \leq H_p \quad \forall p
$$

## 3. Data Structure Examples

### 3.1. Time Slot Example

| Slot ID | Start | End | Length (min) |
| :-- | :-- | :-- | :-- |
| 1 | 09:00 | 09:30 | 30 |
| 2 | 09:30 | 10:00 | 30 |
| ... | ... | ... | ... |

### 3.2. Appointment Example

- Client Mary needs 90 minutes of medication care starting at 09:00:
    - $\mathcal{I}_{\text{need}}(\text{Mary}, \text{medication}, t) = 1$ for $t \in \{\text{09:00}, \text{09:30}, \text{10:00}\}$


### 3.3. Assignment Variable

- $x_{\text{Alice},\ \text{Mary},\ \text{medication},\ 09:00} = 1$ means Alice is assigned to Mary for medication at 09:00–09:30.


## 4. User Interface Considerations

### 4.1. Care Provider (Carer) App

- **Availability Calendar:** Mark available/unavailable slots, holidays, and preferences.
- **Shift Assignments:** View upcoming visits with client/care details.
- **Notifications:** For new assignments or changes.
- **Profile Management:** Skills, preferences, personal info.
- **Leave Management:** Submit/view leave periods.


### 4.2. Client App

- **Care Request Calendar:** Specify care needs, durations, and preferred providers.
- **Booking Overview:** See confirmed and pending bookings.
- **Notifications:** Reminders and updates.
- **Preference Setting:** Indicate preferred providers.


### 4.3. Shared Features

- **Calendar View:** Navigate all scheduled and available slots.
- **Easy Rescheduling:** Request or accept changes.
- **Accessibility:** Large buttons, readable fonts, support for disabilities.


## 5. Current Trends in Care Workforce Scheduling

- **AI-Driven Optimisation:** Leading platforms use AI and advanced algorithms to match staff to shifts, forecast demand, and adapt to last-minute changes[^20_1][^20_2][^20_7].
- **Mobile-First Access:** Both providers and clients expect to manage schedules, requests, and notifications via mobile apps[^20_1][^20_2][^20_6].
- **Compliance \& Credential Tracking:** Automated tracking of certifications, work hours, and regulatory requirements is now standard[^20_2][^20_5].
- **Integrated Communication:** Real-time messaging and alerts are integrated into scheduling platforms for rapid response to changes[^20_6].
- **Self-Service and Flexibility:** Providers can request shifts, swap assignments, and manage their own availability, while clients can specify preferences and recurring needs[^20_1][^20_2][^20_6].
- **Analytics and Reporting:** Platforms offer dashboards and analytics to optimise staffing, monitor compliance, and improve care outcomes[^20_1][^20_3].


## 6. References

- Lieder, A., et al. (2015). "Task scheduling in long-term care facilities: A client-centered approach."
- Morse, A., et al. (2024). "Centralised Scheduling of Nursing Staff: A Rapid Review."
- Koruca, P., et al. (2023). "Personalised staff-scheduling method with work-life balance."
- World Health Organisation (2025). "Health and care workforce planning tools."
- Novagems. "6 Best Staff Scheduling Software For Healthcare in 2025"[^20_1].
- Connecteam. "5 Best Employee Scheduling Software in Australia 2025"[^20_2].
- Moris Media. "Top 10 Healthcare Scheduling Software in Australia for 2025"[^20_3].
- Biarri. "Home Care Scheduling + Rostering Solution Brief"[^20_4].
- Softworks. "Why Automated Scheduling is vital for Care Homes in 2025"[^20_5].
- OnPage. "Best Medical Staff Schedulers of 2025"[^20_6].
- LogicBalls. "The best employee scheduling software in 2025"[^20_7].
- ClickUp. "11 Best Healthcare Scheduling Software Systems in 2025"[^20_8].


## 7. Appendix A: Approaching the Constrained Optimisation Problem

### 7.1. Problem Modeling

- **Define all sets, parameters, and indicator functions** as above.
- **Explicitly encode all operational, regulatory, and preference constraints.**
- **Model appointments as sequences of consecutive slots.**


### 7.2. Solution Approaches

- **Exact Methods:**
Use Mixed-Integer Programming (MIP) or Constraint Programming (CP) for small-to-medium problems with commercial solvers (e.g., Gurobi, CPLEX, Google OR-Tools).
- **Heuristics \& Metaheuristics:**
For large-scale or real-time scheduling, use genetic algorithms, simulated annealing, tabu search, or custom greedy heuristics.
- **Decomposition:**
Break the problem into sub-problems (e.g., first assign slots, then optimise provider-client matching).
- **Rolling Horison:**
Solve for a short window (e.g., 1–2 weeks), then update as new data arrives.
- **AI/ML Augmentation:**
Use machine learning to forecast demand, predict cancellations, or recommend assignments.


### 7.3. Implementation Tips

- **Start with a simple feasible solution, then iteratively improve.**
- **Ensure all data (availability, skills, needs) is up to date and accurate.**
- **Incorporate user feedback loops for continuous improvement.**
- **Test with real and synthetic data to validate robustness.**


## 8. Appendix B: Potential Vendor Solutions

### Open Source Solutions

| Name/Platform | Description | Notes |
| :-- | :-- | :-- |
| **OptaPlanner** | Advanced open-source constraint solver for employee rostering and scheduling | Java-based, highly customisable |
| **Staffjoy** | Open-source scheduling platform (archived, but code available) | Python/Go, web-based |
| **Auto Shift Planner** | Open-source employee shift planning tool | Python, heuristic-based |
| **Cal.com** | Open-source appointment scheduling, can be adapted for workforce needs | Node.js, calendar focus |

### Commercial Solutions

| Name/Platform | Description | Notes |
| :-- | :-- | :-- |
| **Novagems** | All-in-one healthcare scheduling, compliance, and analytics | Time tracking, compliance, mobile |
| **Rotageek** | AI-driven scheduling, demand forecasting, and compliance | Predictive, preference-aware |
| **Smartlinx** | Cloud-based, real-time insights, absence management, payroll | Large-scale, analytics |
| **ShiftCare** | Drag-and-drop scheduling, compliance, recurring shifts | Mobile access, compliance tools[^20_2] |
| **Biarri** | Custom optimisation for home care, Australian regulations | SaaS, global optimisation[^20_4] |
| **AMiON** | Physician scheduling, templates, mobile/web access | Integrates with EHRs[^20_6] |
| **QGenda** | Enterprise-grade, rules-based, analytics, self-service | Credential-aware, mobile[^20_6] |
| **Deputy** | Flexible scheduling, mobile, analytics | Used in healthcare, hospitality[^20_7] |
| **Allocate Software** | Enterprise healthcare scheduling | UK/AU, compliance, analytics |
| **AlayaCare** | Home care management, scheduling, compliance | Cloud-based, care sector specialist |

**Prepared for executive, operational, and technical audiences. For implementation or further technical details, see the technical appendix or contact the author.**



[^20_1]: https://novagems.com/6-best-healthcare-staff-scheduling-software-in-2025/

[^20_2]: https://connecteam.com/best-employee-scheduling-software-australia/

[^20_3]: https://www.morismedia.in/australias-top-10-healthcare-scheduling-software-australia

[^20_4]: https://biarri.com/wp-content/uploads/2025/01/Biarri-Home-Care-Scheduling-Rostering-Solution-Brief.pdf

[^20_5]: https://www.softworks.com/blog/why-automated-scheduling-is-vital-for-care-homes-in-2025/

[^20_6]: https://www.onpage.com/best-medical-staff-schedulers-of-2025/

[^20_7]: https://logicballs.com/app-packs/the-best-employee-scheduling-software-in-2025/

[^20_8]: https://clickup.com/blog/healthcare-scheduling-software-systems/


---

# Appendix: Decision matrix for rating/ranking vendor solutions

## Draft Vendor Decision Matrix for Care Workforce Scheduling Solutions

| Criteria | Weight | Vendor A | Vendor B | Vendor C | Vendor D | Notes/Comments |
| :-- | :--: | :--: | :--: | :--: | :--: | :-- |
| **Core Scheduling Functionality** | 20% |  |  |  |  | Shift/appointment, multi-slot, recurring, split shifts |
| **Healthcare/Care-Specific Features** | 10% |  |  |  |  | Compliance, skill matching, credential tracking |
| **User Experience (UX/UI)** | 10% |  |  |  |  | Intuitive for carers, clients, admins |
| **Mobile Capabilities** | 10% |  |  |  |  | Native apps, responsive design |
| **Integration \& API** | 10% |  |  |  |  | HRIS, payroll, time-tracking, SSO |
| **Scalability \& Performance** | 5% |  |  |  |  | Handles your org’s size, future growth |
| **Customisation \& Flexibility** | 5% |  |  |  |  | Custom fields, workflows, terminology |
| **Compliance \& Regulatory** | 10% |  |  |  |  | Labor laws, union rules, audit trails |
| **Analytics \& Reporting** | 5% |  |  |  |  | Dashboards, export, compliance reports |
| **Support \& Training** | 5% |  |  |  |  | Onboarding, documentation, live help |
| **Implementation \& Onboarding** | 5% |  |  |  |  | Time, resources, migration tools |
| **Cost \& Licensing** | 5% |  |  |  |  | Transparent pricing, TCO, ROI |
| **Vendor Reputation \& Roadmap** | 5% |  |  |  |  | References, innovation, updates |


### Tips for Effective Use

- **Customise weights** based on what matters most for your organisation (e.g., compliance may be weighted higher for healthcare).
- **Involve stakeholders** from IT, operations, frontline staff, and management in scoring[^21_6].
- **Document rationale** for each score to support transparent decision-making.
- **Request demos and reference checks** to validate high scores in critical areas[^21_2][^21_5].

**References:**

- [^21_1] Assembled: How do you evaluate WFM vendors?
- [^21_2] ShiftMed: How to Select the Right Healthcare Workforce Management Vendor
- [^21_5][^21_6][^21_7] myshyft.com: Vendor Selection Criteria and Evaluation Frameworks

This matrix provides a structured, objective foundation for vendor comparison and selection.



[^21_1]: https://www.assembled.com/university/how-do-you-evaluate-wfm-vendors

[^21_2]: https://www.shiftmed.com/insights/knowledge-center/select-the-right-healthcare-workforce-management-vendor/

[^21_3]: https://www.traceconsultants.com.au/thinking/best-practice-rostering-scheduling-for-aged-care-providers-trace-consultants

[^21_4]: https://wfsaustralia.com/healthcare/

[^21_5]: https://www.myshyft.com/blog/vendor-selection-criteria-3/

[^21_6]: https://www.myshyft.com/blog/vendor-evaluation-criteria/

[^21_7]: https://www.myshyft.com/blog/vendor-evaluation-framework/

[^21_8]: https://wfsaustralia.com/guide/the-essential-guide-to-selecting-workforce-management-software/

