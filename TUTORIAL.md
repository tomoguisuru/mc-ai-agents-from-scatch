# Building AI Agents from Scratch

Presenter: Rishabh Misra

## What is Agentic AI?
Intelligent systems that preceives info. processes it to understand context, and autonomously acts to achieve specific objectives, adapting dynamically to changes in its environment.

- **Proactive** Initiate actions addition to responses
- **Goal-oriented** Understands context and map steps to achieve the objective
- **Adaptable** Learn from past experiences, adjust to feedback
- **Autonomy** Make decisions on its own to reach the goals within its parameters

Agents can be though of a LLMs with a set of tools

Generic AI Workflow can be used to generate content on a provided topic
- access required knowledge
- frame coherent answer

Agentic AI can assist in meeting a goal (personal assistant). Will take feedback and refine the outcome.
- thinking
- researching
- set goals
- take actions
- refresh using feedback

## How Agents are Transforming Industires
- Health care
  - enabling personalize treatment
  - processing imaging
- Retail
  - improve customer experience
- Manufacturing
  - improve effeciencies

## Projects: What the market expects

_"GenAI projects are essential for technical screening, especially for candidates without prior ML job experience"_ ~ Appliced Scientist. Amazon

_"AI Projects should show problem framing, creativity, and business relevance, with candidates able to articulate the impat to both technical and non-technical stakeholders"_ ~ Principal MLE, Disney


_"Projects demonstrating proficiency with tools like Docker, TensorFlow, LangChain, and MLOps pipelines are often sufficient for screening for entry and mid level roles"_ ~ Staff MLE, TikTok

Both practical and theoretical knowledge are required for success


## Live Coding: How AI Agents Work
Core Objectives:
- **Implement Natural Language Processing (NLP) -** Accurately interpret and process medical terminology
- **Develop Context-Aware Responses -** Maintain dialogue history for better interaction continuity
- **Enable Multi-Source Integration -** Pull data from reliable medical knowledge bases
- **Provide Real-Time Doctor and Hospital Recommendations -** Offer users relevant medical support
- **Endure Compliance and Ethical AI Usage -** Maintain HIPPA and GDPR compliance

Milestones: End-to-End Agentic AI Demo
- Build Agent 1: Hospital Comparison Agent using a CSV dataset and LangChain's Pandas agent tools
- Build Agent 2: Slot Booking Agent using SQL + LangChain's SQL Toolkit
- Irchestrate both agents using CrewAI with clearly defined tasks, roles, tools, and routing logic
- View Live Demo results with query-based outputs for hospital info, slot booking, and agent coordination

### Step 1: Install dependencies
```bash
pip install httpx langchain langchain_core langchain_openai langchain_experimental crewai
```

### Step 2: Import and Config

```python
import httpx
import pandas as pd