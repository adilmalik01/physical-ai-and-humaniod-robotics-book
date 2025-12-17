// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'introduction',
      label: 'Introduction'
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        {
          type: 'doc',
          id: 'module-1-ros2/index',
          label: 'Overview'
        },
        {
          type: 'doc',
          id: 'module-1-ros2/chapter-01',
          label: 'Chapter 01: ROS 2 Fundamentals'
        },
        {
          type: 'doc',
          id: 'module-1-ros2/chapter-02',
          label: 'Chapter 02: Nodes, Topics, and Services'
        },
        {
          type: 'doc',
          id: 'module-1-ros2/chapter-03',
          label: 'Chapter 03: Real-time Communication Patterns'
        },
        {
          type: 'doc',
          id: 'module-1-ros2/chapter-04',
          label: 'Chapter 04: Bridging AI Logic with ROS 2'
        },
        {
          type: 'doc',
          id: 'module-1-ros2/chapter-05',
          label: 'Chapter 05: Humanoid Body Modeling with URDF'
        }
      ]
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        {
          type: 'doc',
          id: 'module-2-digital-twin/index',
          label: 'Overview'
        },
        {
          type: 'doc',
          id: 'module-2-digital-twin/chapter-01',
          label: 'Chapter 01: Introduction to Digital Twins'
        },
        {
          type: 'doc',
          id: 'module-2-digital-twin/chapter-02',
          label: 'Chapter 02: Gazebo Simulation Fundamentals'
        },
        {
          type: 'doc',
          id: 'module-2-digital-twin/chapter-03',
          label: 'Chapter 03: Creating Humanoid Robot Models for Simulation'
        },
        {
          type: 'doc',
          id: 'module-2-digital-twin/chapter-04',
          label: 'Chapter 04: Physics-Based Environment Design'
        },
        {
          type: 'doc',
          id: 'module-2-digital-twin/chapter-05',
          label: 'Chapter 05: Sensor Simulation and Integration'
        }
      ]
    },
    {
      type: 'category',
      label: 'Module 3: The AI–Robot Brain (NVIDIA Isaac)',
      items: [
        {
          type: 'doc',
          id: 'module-3-ai-robot-brain/index',
          label: 'Overview'
        },
        {
          type: 'doc',
          id: 'module-3-ai-robot-brain/chapter-01',
          label: 'Chapter 01: Introduction to NVIDIA Isaac'
        },
        {
          type: 'doc',
          id: 'module-3-ai-robot-brain/chapter-02',
          label: 'Chapter 02: Isaac Sim for Synthetic Data Generation'
        },
        {
          type: 'doc',
          id: 'module-3-ai-robot-brain/chapter-03',
          label: 'Chapter 03: Isaac ROS for Accelerated Perception'
        },
        {
          type: 'doc',
          id: 'module-3-ai-robot-brain/chapter-04',
          label: 'Chapter 04: Navigation with Nav2 for Humanoids'
        },
        {
          type: 'doc',
          id: 'module-3-ai-robot-brain/chapter-05',
          label: 'Chapter 05: Perception-to-Action Loop Implementation'
        }
      ]
    },
    {
      type: 'category',
      label: 'Module 4: Vision–Language–Action (VLA)',
      items: [
        {
          type: 'doc',
          id: 'module-4-vla/index',
          label: 'Overview'
        },
        {
          type: 'doc',
          id: 'module-4-vla/chapter-01',
          label: 'Chapter 01: Introduction to VLA Systems'
        },
        {
          type: 'doc',
          id: 'module-4-vla/chapter-02',
          label: 'Chapter 02: Speech Recognition with OpenAI Whisper'
        },
        {
          type: 'doc',
          id: 'module-4-vla/chapter-03',
          label: 'Chapter 03: LLM-Based Task Planning'
        },
        {
          type: 'doc',
          id: 'module-4-vla/chapter-04',
          label: 'Chapter 04: Vision-Guided Manipulation'
        },
        {
          type: 'doc',
          id: 'module-4-vla/chapter-05',
          label: 'Chapter 05: End-to-End System Integration'
        }
      ]
    },
    {
      type: 'category',
      label: 'Capstone: Autonomous Humanoid Robot',
      items: [
        {
          type: 'doc',
          id: 'capstone/index',
          label: 'Project Overview'
        },
        {
          type: 'doc',
          id: 'capstone/chapter-01',
          label: 'Chapter 01: System Integration'
        },
        {
          type: 'doc',
          id: 'capstone/chapter-02',
          label: 'Chapter 02: Voice Command to Action Pipeline'
        },
        {
          type: 'doc',
          id: 'capstone/chapter-03',
          label: 'Chapter 03: Autonomous Navigation and Manipulation'
        },
        {
          type: 'doc',
          id: 'capstone/chapter-04',
          label: 'Chapter 04: Testing and Validation'
        }
      ]
    },
    {
      type: 'category',
      label: 'Appendix',
      items: [
        {
          type: 'doc',
          id: 'appendix/index',
          label: 'Resources and References'
        }
      ]
    }
  ]
};

module.exports = sidebars;