import React from 'react';
import Link from '@docusaurus/Link';
import styles from './ModuleCards.module.css';

type ModuleCardType = {
  title: string;
  description: string;
  theme: string;
  link: string;
  icon: string;
};

const ModuleCardList: ModuleCardType[] = [
  {
    title: 'Module 1: The Robotic Nervous System (ROS 2)',
    description: 'Learn ROS 2 fundamentals and real-time communication for humanoid control architecture. Understand nodes, topics, services, and how to bridge Python AI logic with ROS 2.',
    theme: 'Middleware and humanoid control architecture',
    link: '/docs/module-1-ros2',
    icon: '🤖',
  },
  {
    title: 'Module 2: The Digital Twin (Gazebo & Unity)',
    description: 'Explore simulation and human-robot interaction using physics-based environments. Learn to validate humanoid behavior before real-world deployment.',
    theme: 'Simulation and human–robot interaction',
    link: '/docs/module-2-digital-twin',
    icon: '🎮',
  },
  {
    title: 'Module 3: The AI–Robot Brain (NVIDIA Isaac)',
    description: 'Build perception and navigation intelligence using NVIDIA Isaac tools. Learn synthetic data generation and accelerated perception for autonomous navigation.',
    theme: 'Perception and navigation intelligence',
    link: '/docs/module-3-ai-robot-brain',
    icon: '🧠',
  },
  {
    title: 'Module 4: Vision–Language–Action (VLA)',
    description: 'Implement language-driven robotic intelligence with speech-to-action pipelines. Connect LLM-based task planning with vision-guided manipulation.',
    theme: 'Language-driven robotic intelligence',
    link: '/docs/module-4-vla',
    icon: '🗣️',
  },
];

function ModuleCard({ title, description, theme, link, icon }: ModuleCardType) {
  return (
    <div className={styles.moduleCard}>
      <div className={styles.cardHeader}>
        <span className={styles.icon}>{icon}</span>
        <h3 className={styles.cardTitle}>{title}</h3>
      </div>
      <p className={styles.cardTheme}><strong>Theme:</strong> {theme}</p>
      <p className={styles.cardDescription}>{description}</p>
      <div className={styles.cardFooter}>
        <Link className={styles.cardLink} to={link}>
          Explore Module
        </Link>
      </div>
    </div>
  );
}

export default function ModuleCards(): JSX.Element {
  return (
    <section className={styles.modulesSection}>
      <div className="container">
        <h2 className={styles.modulesTitle}>Learning Modules</h2>
        <p className={styles.modulesSubtitle}>Each module represents a logical learning block with practical, real-world applications</p>
        <div className={styles.cardsContainer}>
          {ModuleCardList.map((props, idx) => (
            <ModuleCard key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}