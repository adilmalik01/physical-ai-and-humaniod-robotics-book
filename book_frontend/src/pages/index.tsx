import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import '../css/custom.css';
import { Book, Cpu, Eye, Rocket, Users, BookOpen, Code, Terminal, Brain } from 'lucide-react';




// import './home.css';

export default function Home() {
  return (
    <Layout title="AI & Humanoid Robotics" description=" Master Physical AI & Humanoid Robotics ">


      <div className="home-wrapper">

        <main className="main-content">
          {/* Hero Section */}
          <section className="hero-section">
            {/* <div className="hero-glow"></div> */}
            <div className="hero-container">
              <div className="hero-grid">
                <div className="hero-left">
                  <div className="hero-badge">
                    <Rocket className="badge-icon" />
                    <span className="badge-text">Advanced Humanoid Robotics Training</span>
                  </div>

                  <h1 className="hero-title">
                    Master <span className="gradient-text">Physical AI</span> & Humanoid Robotics
                  </h1>

                  <p className="hero-description">
                    Explore the cutting-edge intersection of artificial intelligence and robotics.
                    Learn to build and control humanoid robots that mimic human form and capabilities.
                  </p>

                  <div className="hero-buttons">
                    <Link href='/docs/introduction' className="btn-primary">
                      Begin Your Journey →
                    </Link>
                    <Link href='/docs/introduction' className="btn-secondary">
                      View Curriculum
                    </Link>
                  </div>

                  <div className="hero-stats">
                    <div className="stat-item">
                      <div className="stat-number stat-blue">4</div>
                      <div className="stat-label">Core Modules</div>
                    </div>
                    <div className="stat-item">
                      <div className="stat-number stat-purple">40+</div>
                      <div className="stat-label">Hours Content</div>
                    </div>
                    <div className="stat-item">
                      <div className="stat-number stat-pink">50+</div>
                      <div className="stat-label">Hands-on Labs</div>
                    </div>
                  </div>
                </div>

                <div className="hero-right">
                  <div className="cards-glow"></div>
                  <div className="cards-container">
                    <div className="feature-card-float card-blue">
                      <div className="card-icon-wrapper icon-blue">
                        <Cpu className="card-icon" />
                      </div>
                      <div>
                        <h3 className="card-title">ROS 2 Framework</h3>
                        <p className="card-subtitle">Robot Operating System</p>
                      </div>
                    </div>

                    <div className="feature-card-float card-purple card-offset">
                      <div className="card-icon-wrapper icon-purple">
                        <Eye className="card-icon" />
                      </div>
                      <div>
                        <h3 className="card-title">Digital Twin</h3>
                        <p className="card-subtitle">Gazebo & Unity Simulation</p>
                      </div>
                    </div>

                    <div className="feature-card-float card-pink">
                      <div className="card-icon-wrapper icon-pink">
                        <Brain className="card-icon" />
                      </div>
                      <div>
                        <h3 className="card-title">VLA Systems</h3>
                        <p className="card-subtitle">Vision-Language-Action AI</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* What You'll Learn Section */}
          <section className="learn-section">
            <div className="section-container">
              <div className="section-header">
                <h2 className="section-title">
                  What You'll <span className="gradient-text">Master</span>
                </h2>
                <p className="section-subtitle">Essential technologies for humanoid robotics development</p>
              </div>

              <div className="modules-grid">
                <div className="module-card module-blue">
                  <div className="module-icon-wrapper icon-blue">
                    <Code className="module-icon" />
                  </div>
                  <h3 className="module-title">ROS 2 Fundamentals</h3>
                  <p className="module-description">Master the robotic nervous system that enables communication and control</p>
                  <ul className="module-list">
                    <li>• Nodes & Topics</li>
                    <li>• Services & Actions</li>
                    <li>• TF2 Transforms</li>
                  </ul>
                </div>

                <div className="module-card module-purple">
                  <div className="module-icon-wrapper icon-purple">
                    <Eye className="module-icon" />
                  </div>
                  <h3 className="module-title">Digital Twin</h3>
                  <p className="module-description">Create accurate virtual representations of physical robots</p>
                  <ul className="module-list">
                    <li>• Gazebo Simulation</li>
                    <li>• Unity Integration</li>
                    <li>• Physics Models</li>
                  </ul>
                </div>

                <div className="module-card module-green">
                  <div className="module-icon-wrapper icon-green">
                    <Cpu className="module-icon" />
                  </div>
                  <h3 className="module-title">NVIDIA Isaac</h3>
                  <p className="module-description">Advanced simulation and AI frameworks for robotics</p>
                  <ul className="module-list">
                    <li>• Isaac Sim</li>
                    <li>• Isaac Gym</li>
                    <li>• GPU Acceleration</li>
                  </ul>
                </div>

                <div className="module-card module-pink">
                  <div className="module-icon-wrapper icon-pink">
                    <Brain className="module-icon" />
                  </div>
                  <h3 className="module-title">VLA Systems</h3>
                  <p className="module-description">Enable robots to perceive, understand, and act on commands</p>
                  <ul className="module-list">
                    <li>• Vision Models</li>
                    <li>• Language Processing</li>
                    <li>• Action Planning</li>
                  </ul>
                </div>
              </div>
            </div>
          </section>

          {/* Target Audience */}
          <section className="audience-section">
            <div className="section-container">
              <div className="section-header">
                <h2 className="section-title">
                  Who Is This <span className="gradient-text-alt">For?</span>
                </h2>
                <p className="section-subtitle">Perfect for professionals and enthusiasts alike</p>
              </div>

              <div className="audience-grid">
                <div className="audience-card">
                  <div className="audience-icon-wrapper">
                    <Cpu className="audience-icon" />
                  </div>
                  <h3 className="audience-title">Robotics Engineers</h3>
                  <p className="audience-desc">Level up your robotics expertise</p>
                </div>

                <div className="audience-card">
                  <div className="audience-icon-wrapper">
                    <Brain className="audience-icon" />
                  </div>
                  <h3 className="audience-title">AI Practitioners</h3>
                  <p className="audience-desc">Apply ML to physical systems</p>
                </div>

                <div className="audience-card">
                  <div className="audience-icon-wrapper">
                    <BookOpen className="audience-icon" />
                  </div>
                  <h3 className="audience-title">Students</h3>
                  <p className="audience-desc">Learn cutting-edge robotics</p>
                </div>

                <div className="audience-card">
                  <div className="audience-icon-wrapper">
                    <Users className="audience-icon" />
                  </div>
                  <h3 className="audience-title">Professionals</h3>
                  <p className="audience-desc">Transition into robotics field</p>
                </div>
              </div>
            </div>
          </section>

          {/* Prerequisites */}
          <section className="prereq-section">
            <div className="prereq-container">
              <div className="section-header">
                <h2 className="section-title">Prerequisites</h2>
                <p className="section-subtitle">What you need to get started</p>
              </div>

              <div className="prereq-grid">
                <div className="prereq-card">
                  <h3 className="prereq-title">
                    <div className="prereq-dot"></div>
                    Programming Knowledge
                  </h3>
                  <ul className="prereq-list">
                    <li><span className="check-icon">✓</span> Python fundamentals</li>
                    <li><span className="check-icon">✓</span> C++ basics</li>
                    <li><span className="check-icon">✓</span> Object-oriented concepts</li>
                  </ul>
                </div>

                <div className="prereq-card">
                  <h3 className="prereq-title">
                    <div className="prereq-dot"></div>
                    Robotics Basics
                  </h3>
                  <ul className="prereq-list">
                    <li><span className="check-icon">✓</span> Fundamental concepts</li>
                    <li><span className="check-icon">✓</span> Kinematics understanding</li>
                    <li><span className="check-icon">✓</span> Control systems</li>
                  </ul>
                </div>

                <div className="prereq-card">
                  <h3 className="prereq-title">
                    <div className="prereq-dot"></div>
                    Linux Skills
                  </h3>
                  <ul className="prereq-list">
                    <li><span className="check-icon">✓</span> Command line proficiency</li>
                    <li><span className="check-icon">✓</span> Package management</li>
                    <li><span className="check-icon">✓</span> File system navigation</li>
                  </ul>
                </div>

                <div className="prereq-card">
                  <h3 className="prereq-title">
                    <div className="prereq-dot"></div>
                    Mathematics
                  </h3>
                  <ul className="prereq-list">
                    <li><span className="check-icon">✓</span> Linear algebra</li>
                    <li><span className="check-icon">✓</span> Calculus basics</li>
                    <li><span className="check-icon">✓</span> Probability & statistics</li>
                  </ul>
                </div>
              </div>
            </div>
          </section>

          {/* CTA Section */}
          <section className="cta-section">
            <div className="cta-glow"></div>
            <div className="cta-container">
              <h2 className="cta-title">
                Ready to Build the <span className="gradient-text">Future?</span>
              </h2>
              <p className="cta-description">
                Start your journey into Physical AI and Humanoid Robotics today.
                Progress from fundamentals to advanced techniques at your own pace.
              </p>
              <div className="cta-buttons">
                <Link href='/docs/introduction' className="btn-primary btn-large">
                  Start Module 1: ROS 2 Fundamentals →
                </Link>

              </div>

              <div className="cta-footer">
                <p>Let's begin this journey into the fascinating world of Physical AI and Humanoid Robotics!</p>
              </div>
            </div>
          </section>

        </main>

        {/* Footer */}
        <footer className="footer">
          <div className="footer-container">
            <p>© 2025 Physical AI & Humanoid Robotics. All rights reserved.</p>
          </div>
        </footer>
      </div>


    </Layout>
  );
}